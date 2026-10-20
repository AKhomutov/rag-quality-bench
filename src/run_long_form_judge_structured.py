"""HISTORICAL EXPERIMENT — not part of normal evaluation flow.

Monolithic gemma3:12b claim-level judge on a fixed candidate (structured output).
Superseded for current long-form work by ``run_split_claim_pipeline``.
Kept for comparison/debug reproducibility.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from claim_level_config import (
    CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
    CLAIM_LEVEL_JUDGE_MODEL,
    CLAIM_LEVEL_JUDGE_MODEL_DIGEST,
    CLAIM_LEVEL_JUDGE_TEMPERATURE,
    CLAIM_LEVEL_NUM_CTX,
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
from clients.ollama_client import OllamaClient
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
JUDGE_TRIALS = 10
JUDGE_GENERATE_TIMEOUT_SECONDS = 900

FAKE_SEVERITY_LABELS = ("Severity 0", "Severity 2.5", "Severity 5")
FAKE_SEVERITY_FACT_HINTS = {
    "Severity 0": ("F6", "F9"),
    "Severity 2.5": ("F7", "F10"),
    "Severity 5": ("F8", "F11"),
}
FAILING_CLAIM_STATUSES = frozenset(
    {
        "contradicted",
        "not_in_evidence",
        "derived_from_unsupported_claim",
    }
)


@dataclass(frozen=True)
class StructuredTrialSummary:
    trial_number: int
    passed: bool
    result: ClaimLevelJudgeResult | None
    invalid: bool
    error: str | None
    parse_error_kind: str | None
    elapsed_seconds: float


def format_verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _count_status(result: ClaimLevelJudgeResult, status: str) -> int:
    return sum(1 for claim in result.claims if claim.support_status == status)


def _count_coverage(result: ClaimLevelJudgeResult, status: str) -> int:
    return sum(1 for item in result.question_coverage if item.status == status)


def _claims_for_label(result: ClaimLevelJudgeResult, label: str) -> list[ClaimAuditItem]:
    lowered = label.lower()
    matches: list[ClaimAuditItem] = []
    for claim in result.claims:
        haystack = f"{claim.claim} {claim.candidate_excerpt}".lower()
        if lowered in haystack:
            matches.append(claim)
    return matches


def fake_severity_correctly_rejected(result: ClaimLevelJudgeResult, label: str) -> bool:
    matches = _claims_for_label(result, label)
    if not matches:
        return False
    if any(claim.support_status == "supported" for claim in matches):
        return False
    return any(claim.support_status in FAILING_CLAIM_STATUSES for claim in matches)


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


def run_judge_trial(
    *,
    ollama: OllamaClient,
    debug_run_dir: Path,
    trial_number: int,
    sut_prompt: str,
    prompt_sha256: str,
    candidate_answer: str,
    candidate_sha256: str,
    authoritative_source: str,
    reference_facts: list[str],
) -> StructuredTrialSummary:
    schema = build_claim_level_json_schema()
    judge_prompt = build_claim_level_judge_prompt(
        authoritative_source=authoritative_source,
        reference_facts=reference_facts,
        original_user_prompt=sut_prompt,
        candidate_answer=candidate_answer,
    )
    started = time.perf_counter()
    try:
        generation = ollama.generate_with_metadata(
            model=CLAIM_LEVEL_JUDGE_MODEL,
            prompt=judge_prompt,
            timeout=JUDGE_GENERATE_TIMEOUT_SECONDS,
            json_schema=schema,
            keep_alive=CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
            temperature=CLAIM_LEVEL_JUDGE_TEMPERATURE,
            num_ctx=CLAIM_LEVEL_NUM_CTX,
        )
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
            num_ctx=CLAIM_LEVEL_NUM_CTX,
            generation=generation,
        )
        elapsed = time.perf_counter() - started
        return StructuredTrialSummary(
            trial_number=trial_number,
            passed=claim_level_passed(result),
            result=result,
            invalid=False,
            error=None,
            parse_error_kind=None,
            elapsed_seconds=elapsed,
        )
    except RuntimeError as exc:
        elapsed = time.perf_counter() - started
        error_text = str(exc)
        return StructuredTrialSummary(
            trial_number=trial_number,
            passed=False,
            result=None,
            invalid=True,
            error=error_text,
            parse_error_kind=classify_parse_error(error_text),
            elapsed_seconds=elapsed,
        )


def print_trial_summary(trial: StructuredTrialSummary) -> None:
    print()
    print(f"=== Judge trial {trial.trial_number} ===")
    print(f"Elapsed: {trial.elapsed_seconds:.1f}s")
    if trial.invalid or trial.result is None:
        print("INVALID / incomplete claim audit")
        print(f"parse_error_kind: {trial.parse_error_kind}")
        print(trial.error)
        return

    result = trial.result
    print(f"Verdict: {format_verdict(trial.passed)}")
    print(f"Total distinct claims audited: {len(result.claims)}")
    print(f"supported: {_count_status(result, 'supported')}")
    print(f"contradicted: {_count_status(result, 'contradicted')}")
    print(f"not_in_evidence: {_count_status(result, 'not_in_evidence')}")
    print(
        f"derived_from_unsupported_claim: {_count_status(result, 'derived_from_unsupported_claim')}"
    )
    provenance_count = sum(1 for claim in result.claims if claim.provenance_issue)
    print(f"provenance issues: {provenance_count}")
    print(
        "coverage: "
        f"answered={_count_coverage(result, 'answered')} "
        f"partial={_count_coverage(result, 'partial')} "
        f"missing={_count_coverage(result, 'missing')}"
    )
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
        print(f"    excerpt: {claim.candidate_excerpt}")
        print(f"    explanation: {claim.explanation}")

    inconsistencies = find_internal_inconsistencies(result)
    print(f"internal semantic inconsistencies: {len(inconsistencies)}")
    for item in inconsistencies:
        print(f"  - {item}")

    if not trial.passed:
        print("Failure reasons:")
        for reason in claim_level_failure_reasons(result):
            print(f"  - {reason}")

    print("Rationale:")
    print(result.rationale)


def print_overall_summary(
    trials: list[StructuredTrialSummary],
    *,
    total_wall_seconds: float,
) -> None:
    valid = [trial for trial in trials if not trial.invalid]
    invalid = [trial for trial in trials if trial.invalid]
    passed_count = sum(1 for trial in valid if trial.passed)
    failed_count = len(valid) - passed_count

    print()
    print("=== Structured-output claim-level summary (gemma3:12b) ===")
    print()
    print(f"Judge model: {CLAIM_LEVEL_JUDGE_MODEL}")
    print(f"Judge model digest: {CLAIM_LEVEL_JUDGE_MODEL_DIGEST}")
    print(f"keep_alive: {CLAIM_LEVEL_JUDGE_KEEP_ALIVE}")
    print()
    print(f"Valid structured outputs: {len(valid)}/10")
    print(f"Invalid outputs: {len(invalid)}/10")
    print()
    print("Parse error kinds:")
    kinds: dict[str, int] = {}
    for trial in invalid:
        kind = trial.parse_error_kind or "unknown"
        kinds[kind] = kinds.get(kind, 0) + 1
    if not kinds:
        print("  (none)")
    else:
        for kind, count in sorted(kinds.items()):
            print(f"  {kind}: {count}")

    if valid:
        dominant_pass = passed_count >= failed_count
        dominant_count = max(passed_count, failed_count)
        stability = dominant_count / len(valid)
        print()
        print(f"PASS: {passed_count}")
        print(f"FAIL: {failed_count}")
        print(f"Verdict stability (valid trials): {stability * 100.0:.1f}%")
        print(f"Dominant verdict: {format_verdict(dominant_pass)}")

        print()
        print("Fake severity rejection diagnostics:")
        for label in FAKE_SEVERITY_LABELS:
            rejected = sum(
                1
                for trial in valid
                if trial.result is not None
                and fake_severity_correctly_rejected(trial.result, label)
            )
            print(f"  {label} correctly rejected: {rejected}/{len(valid)}")

    elapsed_values = [trial.elapsed_seconds for trial in trials]
    average = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0
    print()
    print("Performance:")
    print(f"  total wall time: {total_wall_seconds:.1f}s")
    print(f"  average per trial: {average:.1f}s")

    print()
    print("=== qwen3:8b vs gemma3:12b ===")
    print()
    print("qwen3:8b (previous structured baseline):")
    print("  Schema validity: 10/10")
    print("  PASS 1 / FAIL 9")
    print("  Fake Severity 0/2.5/5: mostly supported")
    print("  contradicted: 0")
    print("  derived_from_unsupported_claim: 0")
    print("  provenance_issue: 0")
    print("  Trial 10: false PASS")
    print()
    print("gemma3:12b (this run):")
    print(f"  Schema validity: {len(valid)}/10")
    print(f"  PASS {passed_count} / FAIL {failed_count}")
    if valid:
        for label in FAKE_SEVERITY_LABELS:
            rejected = sum(
                1
                for trial in valid
                if trial.result is not None
                and fake_severity_correctly_rejected(trial.result, label)
            )
            print(f"  {label} correctly rejected: {rejected}/{len(valid)}")
        contradicted_total = sum(
            _count_status(trial.result, "contradicted")
            for trial in valid
            if trial.result is not None
        )
        derived_total = sum(
            _count_status(trial.result, "derived_from_unsupported_claim")
            for trial in valid
            if trial.result is not None
        )
        provenance_total = sum(
            sum(1 for claim in trial.result.claims if claim.provenance_issue)
            for trial in valid
            if trial.result is not None
        )
        print(f"  contradicted claim instances (sum over trials): {contradicted_total}")
        print(f"  derived_from_unsupported_claim instances: {derived_total}")
        print(f"  provenance_issue instances: {provenance_total}")


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
    print("SUT was NOT called. Reusing fixed candidate answer from previous trace.")
    print()
    print("Semantic prompt / F1-F16 / JSON Schema / Python verdict: unchanged.")
    print(f"Judge model change only: qwen3:8b -> {CLAIM_LEVEL_JUDGE_MODEL}")
    print(f"Judge model digest: {CLAIM_LEVEL_JUDGE_MODEL_DIGEST}")
    print(f"keep_alive: {CLAIM_LEVEL_JUDGE_KEEP_ALIVE}")
    print()
    print(f"Schema version: {CLAIM_LEVEL_SCHEMA_VERSION}")
    print()
    print("Prompt fixture:")
    print(PROMPT_FIXTURE_RELATIVE_PATH)
    print()
    print("Prompt SHA-256:")
    print(prompt_sha256)
    print()
    print(f"Judge debug traces:\n{debug_run_dir}")
    print()

    ollama = OllamaClient()
    trials: list[StructuredTrialSummary] = []
    wall_started = time.perf_counter()
    for trial_number in range(1, JUDGE_TRIALS + 1):
        print(
            f"Running {CLAIM_LEVEL_JUDGE_MODEL} claim-level judge trial "
            f"{trial_number}/{JUDGE_TRIALS}..."
        )
        trial = run_judge_trial(
            ollama=ollama,
            debug_run_dir=debug_run_dir,
            trial_number=trial_number,
            sut_prompt=sut_prompt,
            prompt_sha256=prompt_sha256,
            candidate_answer=candidate_answer,
            candidate_sha256=candidate_sha256,
            authoritative_source=authoritative_source,
            reference_facts=reference_facts,
        )
        trials.append(trial)
        print_trial_summary(trial)
    total_wall = time.perf_counter() - wall_started

    print_overall_summary(trials, total_wall_seconds=total_wall)
    print()
    print(f"Judge debug traces:\n{debug_run_dir / f'{CASE_ID}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
