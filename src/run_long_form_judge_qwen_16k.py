"""HISTORICAL EXPERIMENT — A/B control; not part of normal evaluation flow.

Single-trial Qwen monolithic claim-level judge with num_ctx=16384.
Compares against the previous default-~4k Qwen baseline and the Gemma 16K trial.
Current long-form work uses ``run_split_claim_pipeline`` instead.
"""

from __future__ import annotations

import time
from typing import cast

from claim_level_config import (
    CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
    CLAIM_LEVEL_JUDGE_TEMPERATURE,
    CLAIM_LEVEL_NUM_CTX,
    EXPECTED_FIXED_CANDIDATE_LENGTH,
    EXPECTED_FIXED_CANDIDATE_SHA256,
    SUT_GENERATION_MODEL,
    SUT_GENERATION_MODEL_DIGEST,
)
from claim_level_debug import parse_and_trace_claim_level_judge_response
from claim_level_diagnostics import classify_parse_error
from claim_level_results import (
    ClaimAuditItem,
    ClaimLevelJudgeResult,
    claim_level_failure_reasons,
    claim_level_passed,
)
from clients.ollama_client import OllamaClient, build_generate_request_body
from fixed_candidate_loader import load_identical_candidate_answer, sha256_text
from judge_debug import case_trace_path, create_judge_debug_run_dir
from judges.claim_level_judge import build_claim_level_judge_prompt
from judges.claim_level_schema import (
    CLAIM_LEVEL_SCHEMA_VERSION,
    build_claim_level_json_schema,
)
from long_form_severity_facts import reference_facts_as_lines
from long_form_severity_prompt import (
    AUTHORITATIVE_DOCUMENT_PATH,
    CASE_ID,
    PROMPT_FIXTURE_RELATIVE_PATH,
    assert_prompt_fixture_integrity,
    load_sut_prompt,
)
from workspace_setup import PROJECT_ROOT

REPORTS_ROOT = PROJECT_ROOT / "reports"
PREVIOUS_TRACE_PATH = (
    REPORTS_ROOT / "judge_debug" / "20261019T235938" / "mixed_truth_severity_policy_poisoning.jsonl"
)
JUDGE_GENERATE_TIMEOUT_SECONDS = 900

FAKE_SEVERITY_LABELS = ("Severity 0", "Severity 2.5", "Severity 5")
FAKE_SEVERITY_FACT_HINTS = {
    "Severity 0": ("F6", "F9"),
    "Severity 2.5": ("F7", "F10"),
    "Severity 5": ("F8", "F11"),
}


def format_verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def assert_fixed_candidate_identity(candidate_answer: str, candidate_sha256: str) -> None:
    if candidate_sha256 != EXPECTED_FIXED_CANDIDATE_SHA256:
        msg = (
            "candidate SHA-256 mismatch: "
            f"expected {EXPECTED_FIXED_CANDIDATE_SHA256}, got {candidate_sha256}"
        )
        raise ValueError(msg)
    if len(candidate_answer) != EXPECTED_FIXED_CANDIDATE_LENGTH:
        msg = (
            "candidate length mismatch: "
            f"expected {EXPECTED_FIXED_CANDIDATE_LENGTH}, got {len(candidate_answer)}"
        )
        raise ValueError(msg)


def _count_status(result: ClaimLevelJudgeResult, status: str) -> int:
    return sum(1 for claim in result.claims if claim.support_status == status)


def _claims_for_label(result: ClaimLevelJudgeResult, label: str) -> list[ClaimAuditItem]:
    lowered = label.lower()
    return [
        claim
        for claim in result.claims
        if lowered in f"{claim.claim} {claim.candidate_excerpt}".lower()
    ]


def find_internal_inconsistencies(result: ClaimLevelJudgeResult) -> list[str]:
    findings: list[str] = []
    for claim in result.claims:
        haystack = f"{claim.claim} {claim.candidate_excerpt}".lower()
        for label, fact_ids in FAKE_SEVERITY_FACT_HINTS.items():
            if label.lower() not in haystack:
                continue
            if claim.support_status != "supported":
                continue
            overlapping = [fact_id for fact_id in claim.reference_fact_ids if fact_id in fact_ids]
            if overlapping:
                findings.append(
                    f"supported + {overlapping} while discussing {label}: {claim.claim}"
                )
    return findings


def print_semantic(result: ClaimLevelJudgeResult, passed: bool) -> None:
    print()
    print(f"semantic verdict: {format_verdict(passed)}")
    print(f"claims: {len(result.claims)}")
    print(f"supported: {_count_status(result, 'supported')}")
    print(f"contradicted: {_count_status(result, 'contradicted')}")
    print(f"not_in_evidence: {_count_status(result, 'not_in_evidence')}")
    print(
        f"derived_from_unsupported_claim: {_count_status(result, 'derived_from_unsupported_claim')}"
    )
    provenance_items = [claim for claim in result.claims if claim.provenance_issue]
    print(f"provenance issues: {len(provenance_items)}")
    answered = sum(1 for item in result.question_coverage if item.status == "answered")
    partial = sum(1 for item in result.question_coverage if item.status == "partial")
    missing = sum(1 for item in result.question_coverage if item.status == "missing")
    print(f"coverage: answered={answered} partial={partial} missing={missing}")
    print(f"final_table_present: {result.final_table_present}")
    print(f"final_summary_present: {result.final_summary_present}")

    for label in FAKE_SEVERITY_LABELS:
        matches = _claims_for_label(result, label)
        print(f"Claims involving {label}: {len(matches)}")
        for claim in matches:
            print(f"  claim: {claim.claim}")
            print(f"  support_status: {claim.support_status}")
            print(f"  reference_fact_ids: {claim.reference_fact_ids}")
            print(f"  provenance_issue: {claim.provenance_issue}")
            print(f"  explanation: {claim.explanation}")

    derived = [
        claim for claim in result.claims if claim.support_status == "derived_from_unsupported_claim"
    ]
    print(f"derived_from_unsupported_claim findings: {len(derived)}")
    for claim in derived:
        print(f"  - {claim.claim}")
        print(f"    explanation: {claim.explanation}")

    print(f"provenance_issue=true items: {len(provenance_items)}")
    for claim in provenance_items:
        print(f"  - [{claim.support_status}] {claim.claim}")

    inconsistencies = find_internal_inconsistencies(result)
    print(f"internal semantic inconsistencies: {len(inconsistencies)}")
    for item in inconsistencies:
        print(f"  - {item}")

    if not passed:
        print("Failure reasons:")
        for reason in claim_level_failure_reasons(result):
            print(f"  - {reason}")
    print("Rationale:")
    print(result.rationale)


def print_comparison_table(
    *,
    schema_valid: bool,
    done_reason: str | None,
    result: ClaimLevelJudgeResult | None,
    passed: bool | None,
) -> None:
    if result is None or passed is None:
        fake = "n/a"
        derived = "n/a"
        provenance = "n/a"
        verdict = "n/a (invalid JSON)"
    else:
        fake_statuses: list[str] = []
        for label in FAKE_SEVERITY_LABELS:
            matches = _claims_for_label(result, label)
            if not matches:
                fake_statuses.append(f"{label}:none")
            else:
                statuses = sorted({claim.support_status for claim in matches})
                fake_statuses.append(f"{label}:{','.join(statuses)}")
        fake = "; ".join(fake_statuses)
        derived = str(_count_status(result, "derived_from_unsupported_claim"))
        provenance = str(sum(1 for claim in result.claims if claim.provenance_issue))
        verdict = format_verdict(passed)

    print()
    print("=== Comparison table ===")
    print()
    print(f"{'Metric':<28} {'qwen old':<22} {'qwen 16K':<22} {'gemma 16K':<22}")
    print(f"{'Schema valid':<28} {'10/10':<22} {'yes' if schema_valid else 'no':<22} {'yes':<22}")
    print(f"{'done_reason':<28} {'(not recorded)':<22} {done_reason!s:<22} {'stop':<22}")
    print(
        f"{'Fake severity handling':<28} {'mostly supported':<22} "
        f"{fake[:22]:<22} {'mostly not_in_ev+prov':<22}"
    )
    print(f"{'Derived claims':<28} {'0':<22} {derived:<22} {'5':<22}")
    print(f"{'Provenance issues':<28} {'0':<22} {provenance:<22} {'14':<22}")
    print(f"{'Verdict':<28} {'PASS1/FAIL9':<22} {verdict:<22} {'FAIL':<22}")


def main() -> int:
    prompt_sha256 = assert_prompt_fixture_integrity()
    sut_prompt = load_sut_prompt()
    authoritative_source = AUTHORITATIVE_DOCUMENT_PATH.read_text(encoding="utf-8")
    reference_facts = reference_facts_as_lines()
    candidate_answer = load_identical_candidate_answer(PREVIOUS_TRACE_PATH)
    candidate_sha256 = sha256_text(candidate_answer)
    assert_fixed_candidate_identity(candidate_answer, candidate_sha256)
    debug_run_dir = create_judge_debug_run_dir(REPORTS_ROOT)

    print("Candidate source:")
    print(PREVIOUS_TRACE_PATH.relative_to(PROJECT_ROOT))
    print()
    print("Candidate SHA-256:")
    print(candidate_sha256)
    print()
    print("Candidate length:")
    print(len(candidate_answer))
    print()
    print("SUT was NOT called.")
    print("Semantic prompt / F1-F16 / schema / verdict unchanged.")
    print(f"Judge model: {SUT_GENERATION_MODEL}")
    print(f"Judge digest: {SUT_GENERATION_MODEL_DIGEST}")
    print(f"num_ctx: {CLAIM_LEVEL_NUM_CTX}")
    print(f"temperature: {CLAIM_LEVEL_JUDGE_TEMPERATURE}")
    print(f"keep_alive: {CLAIM_LEVEL_JUDGE_KEEP_ALIVE}")
    print(f"schema version: {CLAIM_LEVEL_SCHEMA_VERSION}")
    print()
    print(f"Judge debug traces:\n{debug_run_dir}")
    print()

    schema = build_claim_level_json_schema()
    judge_prompt = build_claim_level_judge_prompt(
        authoritative_source=authoritative_source,
        reference_facts=reference_facts,
        original_user_prompt=sut_prompt,
        candidate_answer=candidate_answer,
    )
    request_body = build_generate_request_body(
        model=SUT_GENERATION_MODEL,
        prompt=judge_prompt,
        json_schema=schema,
        keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
        temperature=CLAIM_LEVEL_JUDGE_TEMPERATURE,
        num_ctx=CLAIM_LEVEL_NUM_CTX,
    )
    options_raw = request_body["options"]
    assert isinstance(options_raw, dict)
    request_options = cast(dict[str, object], options_raw)
    print(f"request options: {request_options}")
    print()

    ollama = OllamaClient()
    started = time.perf_counter()
    generation = ollama.generate_with_metadata(
        model=SUT_GENERATION_MODEL,
        prompt=judge_prompt,
        timeout=JUDGE_GENERATE_TIMEOUT_SECONDS,
        json_schema=schema,
        keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
        temperature=CLAIM_LEVEL_JUDGE_TEMPERATURE,
        num_ctx=CLAIM_LEVEL_NUM_CTX,
    )
    wall = time.perf_counter() - started

    print("=== Completion metadata ===")
    print(f"done: {generation.done}")
    print(f"done_reason: {generation.done_reason}")
    print(f"prompt_eval_count: {generation.prompt_eval_count}")
    print(f"eval_count: {generation.eval_count}")
    print(f"total_duration_ns: {generation.total_duration}")
    print(f"load_duration_ns: {generation.load_duration}")
    print(f"prompt_eval_duration_ns: {generation.prompt_eval_duration}")
    print(f"eval_duration_ns: {generation.eval_duration}")
    print(f"raw response chars: {len(generation.response)}")
    print(f"wall time: {wall:.1f}s")
    print()
    print("Gemma 16K prompt_eval_count reference: 4799")
    print()

    schema_valid = False
    result: ClaimLevelJudgeResult | None = None
    passed: bool | None = None
    try:
        result = parse_and_trace_claim_level_judge_response(
            path=case_trace_path(debug_run_dir, CASE_ID),
            case_id=CASE_ID,
            trial_number=1,
            prompt_fixture_path=PROMPT_FIXTURE_RELATIVE_PATH,
            prompt_sha256=prompt_sha256,
            sut_prompt=sut_prompt,
            candidate_answer=candidate_answer,
            candidate_sha256=candidate_sha256,
            authoritative_source=authoritative_source,
            reference_facts=reference_facts,
            judge_prompt=judge_prompt,
            raw_response=generation.response,
            model=SUT_GENERATION_MODEL,
            model_digest=SUT_GENERATION_MODEL_DIGEST,
            temperature=float(CLAIM_LEVEL_JUDGE_TEMPERATURE),
            keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
            claim_level_schema_version=CLAIM_LEVEL_SCHEMA_VERSION,
            num_ctx=CLAIM_LEVEL_NUM_CTX,
            generation=generation,
        )
        schema_valid = True
        passed = claim_level_passed(result)
        print("parse valid: True")
        print_semantic(result, passed)
    except RuntimeError as exc:
        error_text = str(exc)
        print("parse valid: False")
        print(f"parse_error_kind: {classify_parse_error(error_text)}")
        print(f"parse error (head): {error_text.splitlines()[0]}")

    print_comparison_table(
        schema_valid=schema_valid,
        done_reason=generation.done_reason,
        result=result,
        passed=passed,
    )
    print()
    print(f"Judge debug traces:\n{debug_run_dir / f'{CASE_ID}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
