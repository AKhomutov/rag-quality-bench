"""HISTORICAL EXPERIMENT — diagnostic only; not part of normal evaluation flow.

Inspects gemma3:12b truncation / num_ctx behaviour on a fixed candidate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from claim_level_config import (
    CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
    CLAIM_LEVEL_JUDGE_MODEL,
    CLAIM_LEVEL_JUDGE_MODEL_DIGEST,
    CLAIM_LEVEL_JUDGE_TEMPERATURE,
    EXPECTED_FIXED_CANDIDATE_LENGTH,
    EXPECTED_FIXED_CANDIDATE_SHA256,
)
from claim_level_debug import parse_and_trace_claim_level_judge_response
from claim_level_diagnostics import classify_parse_error
from claim_level_results import (
    ClaimAuditItem,
    ClaimLevelJudgeResult,
    claim_level_failure_reasons,
    claim_level_passed,
)
from clients.ollama_client import (
    OllamaClient,
    OllamaGenerationResult,
    build_generate_request_body,
)
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
DIAGNOSTIC_NUM_CTX = 16384

FAKE_SEVERITY_LABELS = ("Severity 0", "Severity 2.5", "Severity 5")
FAKE_SEVERITY_FACT_HINTS = {
    "Severity 0": ("F6", "F9"),
    "Severity 2.5": ("F7", "F10"),
    "Severity 5": ("F8", "F11"),
}


@dataclass(frozen=True)
class DiagnosticCallResult:
    label: str
    num_ctx: int | None
    generation: OllamaGenerationResult
    wall_seconds: float
    parse_valid: bool
    parse_error_kind: str | None
    parse_error: str | None
    result: ClaimLevelJudgeResult | None
    passed: bool | None
    request_options: dict[str, object]


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


def inspect_model_context(ollama: OllamaClient) -> None:
    print("=== Local model / context inspection ===")
    print()
    print(f"Model: {CLAIM_LEVEL_JUDGE_MODEL}")
    print(f"Pinned digest: {CLAIM_LEVEL_JUDGE_MODEL_DIGEST}")
    print()
    show = ollama.show_model(CLAIM_LEVEL_JUDGE_MODEL)
    details = show.get("details")
    if isinstance(details, dict):
        details_obj = cast(dict[str, object], details)
        print(f"show.details.family: {details_obj.get('family')}")
        print(f"show.details.parameter_size: {details_obj.get('parameter_size')}")
    model_info = show.get("model_info")
    if isinstance(model_info, dict):
        info = cast(dict[str, object], model_info)
        interesting_keys = [
            key
            for key in info
            if "context" in key.lower() or "ctx" in key.lower() or key.endswith(".context_length")
        ]
        if interesting_keys:
            print("show.model_info context-related fields:")
            for key in sorted(interesting_keys):
                print(f"  {key}: {info[key]}")
        else:
            print("show.model_info: no context-related keys found")
    parameters = show.get("parameters")
    if isinstance(parameters, str) and parameters.strip():
        print("show.parameters:")
        print(parameters)
    print()


def truncation_justifies_16k_retry(call: DiagnosticCallResult) -> bool:
    generation = call.generation
    reason = (generation.done_reason or "").lower()
    length_ceiling = reason in {"length", "max_tokens", "max tokens"}
    unterminated_json = call.parse_error_kind == "invalid_json" and "Unterminated string" in (
        call.parse_error or ""
    )
    # Completed generation but truncated JSON can still indicate output ceiling.
    large_truncated_stop = (
        call.parse_error_kind == "invalid_json"
        and generation.done is True
        and reason == "stop"
        and generation.eval_count is not None
        and generation.eval_count >= 1000
    )
    return length_ceiling or unterminated_json or large_truncated_stop


def classify_limit_category(call: DiagnosticCallResult) -> str:
    generation = call.generation
    reason = (generation.done_reason or "").lower()
    if reason in {"length", "max_tokens", "max tokens"}:
        return "generation/output token limit (done_reason indicates length ceiling)"
    if call.parse_error_kind == "invalid_json" and "Unterminated string" in (
        call.parse_error or ""
    ):
        if generation.prompt_eval_count is not None and generation.eval_count is not None:
            return (
                "likely generation/output truncation "
                f"(unterminated JSON; prompt_eval_count={generation.prompt_eval_count}, "
                f"eval_count={generation.eval_count}, done_reason={generation.done_reason!r})"
            )
        return "unterminated JSON; cannot cleanly separate prompt vs generation ceiling"
    if reason == "stop" and call.parse_valid:
        return "normal completion (done_reason=stop)"
    if reason == "stop" and not call.parse_valid:
        return (
            "other termination with done_reason=stop but invalid parse "
            "(not clearly labeled as length by Ollama)"
        )
    return f"other termination reason (done_reason={generation.done_reason!r})"


def run_diagnostic_call(
    *,
    label: str,
    ollama: OllamaClient,
    debug_run_dir: Path,
    trial_number: int,
    sut_prompt: str,
    prompt_sha256: str,
    candidate_answer: str,
    candidate_sha256: str,
    authoritative_source: str,
    reference_facts: list[str],
    num_ctx: int | None,
) -> DiagnosticCallResult:
    schema = build_claim_level_json_schema()
    judge_prompt = build_claim_level_judge_prompt(
        authoritative_source=authoritative_source,
        reference_facts=reference_facts,
        original_user_prompt=sut_prompt,
        candidate_answer=candidate_answer,
    )
    request_body = build_generate_request_body(
        model=CLAIM_LEVEL_JUDGE_MODEL,
        prompt=judge_prompt,
        json_schema=schema,
        keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
        temperature=CLAIM_LEVEL_JUDGE_TEMPERATURE,
        num_ctx=num_ctx,
    )
    options_raw = request_body["options"]
    assert isinstance(options_raw, dict)
    request_options = cast(dict[str, object], options_raw)

    started = time.perf_counter()
    generation = ollama.generate_with_metadata(
        model=CLAIM_LEVEL_JUDGE_MODEL,
        prompt=judge_prompt,
        timeout=JUDGE_GENERATE_TIMEOUT_SECONDS,
        json_schema=schema,
        keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
        temperature=CLAIM_LEVEL_JUDGE_TEMPERATURE,
        num_ctx=num_ctx,
    )
    wall = time.perf_counter() - started

    try:
        result = parse_and_trace_claim_level_judge_response(
            path=case_trace_path(debug_run_dir, CASE_ID),
            case_id=CASE_ID,
            trial_number=trial_number,
            prompt_fixture_path=PROMPT_FIXTURE_RELATIVE_PATH,
            prompt_sha256=prompt_sha256,
            sut_prompt=sut_prompt,
            candidate_answer=candidate_answer,
            candidate_sha256=candidate_sha256,
            authoritative_source=authoritative_source,
            reference_facts=reference_facts,
            judge_prompt=judge_prompt,
            raw_response=generation.response,
            model=CLAIM_LEVEL_JUDGE_MODEL,
            model_digest=CLAIM_LEVEL_JUDGE_MODEL_DIGEST,
            temperature=float(CLAIM_LEVEL_JUDGE_TEMPERATURE),
            keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
            claim_level_schema_version=CLAIM_LEVEL_SCHEMA_VERSION,
            num_ctx=num_ctx,
            generation=generation,
        )
        return DiagnosticCallResult(
            label=label,
            num_ctx=num_ctx,
            generation=generation,
            wall_seconds=wall,
            parse_valid=True,
            parse_error_kind=None,
            parse_error=None,
            result=result,
            passed=claim_level_passed(result),
            request_options=request_options,
        )
    except RuntimeError as exc:
        error_text = str(exc)
        return DiagnosticCallResult(
            label=label,
            num_ctx=num_ctx,
            generation=generation,
            wall_seconds=wall,
            parse_valid=False,
            parse_error_kind=classify_parse_error(error_text),
            parse_error=error_text,
            result=None,
            passed=None,
            request_options=request_options,
        )


def print_call_report(call: DiagnosticCallResult) -> None:
    generation = call.generation
    print()
    print(f"=== Diagnostic call: {call.label} ===")
    print()
    print(f"model: {CLAIM_LEVEL_JUDGE_MODEL}")
    print(f"digest: {CLAIM_LEVEL_JUDGE_MODEL_DIGEST}")
    if call.num_ctx is None:
        print("num_ctx request value: absent")
    else:
        print(f"num_ctx request value: {call.num_ctx}")
    print(f"request options: {call.request_options}")
    print(f"keep_alive: {CLAIM_LEVEL_JUDGE_KEEP_ALIVE}")
    print(f"temperature: {CLAIM_LEVEL_JUDGE_TEMPERATURE}")
    print(f"schema version: {CLAIM_LEVEL_SCHEMA_VERSION}")
    print()
    print(f"done: {generation.done}")
    print(f"done_reason: {generation.done_reason}")
    print(f"prompt_eval_count: {generation.prompt_eval_count}")
    print(f"eval_count: {generation.eval_count}")
    print(f"total_duration_ns: {generation.total_duration}")
    print(f"load_duration_ns: {generation.load_duration}")
    print(f"prompt_eval_duration_ns: {generation.prompt_eval_duration}")
    print(f"eval_duration_ns: {generation.eval_duration}")
    print(f"raw response chars: {len(generation.response)}")
    print(f"wall time: {call.wall_seconds:.1f}s")
    print()
    print(f"parse valid: {call.parse_valid}")
    print(f"parse_error_kind: {call.parse_error_kind}")
    print(f"limit category: {classify_limit_category(call)}")
    if call.parse_error is not None:
        first_line = call.parse_error.splitlines()[0]
        print(f"parse error (head): {first_line}")

    if call.result is None or call.passed is None:
        return

    result = call.result
    print()
    print(f"semantic verdict: {format_verdict(call.passed)}")
    print(f"claims: {len(result.claims)}")
    print(f"supported: {_count_status(result, 'supported')}")
    print(f"contradicted: {_count_status(result, 'contradicted')}")
    print(f"not_in_evidence: {_count_status(result, 'not_in_evidence')}")
    print(
        f"derived_from_unsupported_claim: {_count_status(result, 'derived_from_unsupported_claim')}"
    )
    provenance = sum(1 for claim in result.claims if claim.provenance_issue)
    print(f"provenance issues: {provenance}")
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

    inconsistencies = find_internal_inconsistencies(result)
    print(f"internal semantic inconsistencies: {len(inconsistencies)}")
    for item in inconsistencies:
        print(f"  - {item}")

    if not call.passed:
        print("Failure reasons:")
        for reason in claim_level_failure_reasons(result):
            print(f"  - {reason}")
    print("Rationale:")
    print(result.rationale)


def print_comparison(
    current: DiagnosticCallResult,
    retry: DiagnosticCallResult | None,
) -> None:
    print()
    print("=== current vs 16384 ===")
    print()
    print("current:")
    print("  num_ctx: absent")
    print(f"  prompt_eval_count: {current.generation.prompt_eval_count}")
    print(f"  eval_count: {current.generation.eval_count}")
    print(f"  done/done_reason: {current.generation.done}/{current.generation.done_reason}")
    print(f"  raw chars: {len(current.generation.response)}")
    print(f"  parse valid: {current.parse_valid} ({current.parse_error_kind})")
    print(f"  wall: {current.wall_seconds:.1f}s")
    if retry is None:
        print()
        print("16384 retry: not run")
        return
    print()
    print("num_ctx=16384:")
    print(f"  prompt_eval_count: {retry.generation.prompt_eval_count}")
    print(f"  eval_count: {retry.generation.eval_count}")
    print(f"  done/done_reason: {retry.generation.done}/{retry.generation.done_reason}")
    print(f"  raw chars: {len(retry.generation.response)}")
    print(f"  parse valid: {retry.parse_valid} ({retry.parse_error_kind})")
    print(f"  wall: {retry.wall_seconds:.1f}s")


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
    print(f"Judge debug traces:\n{debug_run_dir}")
    print()

    ollama = OllamaClient()
    inspect_model_context(ollama)

    current = run_diagnostic_call(
        label="current settings (num_ctx absent)",
        ollama=ollama,
        debug_run_dir=debug_run_dir,
        trial_number=1,
        sut_prompt=sut_prompt,
        prompt_sha256=prompt_sha256,
        candidate_answer=candidate_answer,
        candidate_sha256=candidate_sha256,
        authoritative_source=authoritative_source,
        reference_facts=reference_facts,
        num_ctx=None,
    )
    print_call_report(current)

    retry: DiagnosticCallResult | None = None
    if truncation_justifies_16k_retry(current):
        print()
        print(
            f"16K retry justified by: {classify_limit_category(current)}. "
            f"Running one additional call with num_ctx={DIAGNOSTIC_NUM_CTX}."
        )
        retry = run_diagnostic_call(
            label=f"num_ctx={DIAGNOSTIC_NUM_CTX}",
            ollama=ollama,
            debug_run_dir=debug_run_dir,
            trial_number=2,
            sut_prompt=sut_prompt,
            prompt_sha256=prompt_sha256,
            candidate_answer=candidate_answer,
            candidate_sha256=candidate_sha256,
            authoritative_source=authoritative_source,
            reference_facts=reference_facts,
            num_ctx=DIAGNOSTIC_NUM_CTX,
        )
        print_call_report(retry)
    else:
        print()
        print("16K retry not justified by first-call metadata/parse outcome.")

    print_comparison(current, retry)
    print()
    print(f"Judge debug traces:\n{debug_run_dir / f'{CASE_ID}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
