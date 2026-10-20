"""HISTORICAL EXPERIMENT — not part of normal evaluation flow.

Live SUT once + monolithic gemma3:12b claim-level judge trials.
Superseded for current long-form work by ``run_split_claim_pipeline``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from claim_level_config import (
    CLAIM_LEVEL_JUDGE_KEEP_ALIVE,
    CLAIM_LEVEL_JUDGE_MODEL,
    CLAIM_LEVEL_JUDGE_MODEL_DIGEST,
    CLAIM_LEVEL_JUDGE_TEMPERATURE,
    CLAIM_LEVEL_NUM_CTX,
)
from claim_level_debug import parse_and_trace_claim_level_judge_response
from claim_level_results import (
    ClaimLevelJudgeResult,
    claim_level_failure_reasons,
    claim_level_passed,
)
from clients.anythingllm_client import AnythingLLMClient
from clients.ollama_client import OllamaClient
from fixed_candidate_loader import sha256_text
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
from workspace_setup import (
    PROJECT_ROOT,
    PreparedWorkspace,
    cleanup_workspace,
    prepare_workspace,
)

REPORTS_ROOT = PROJECT_ROOT / "reports"
JUDGE_TRIALS = 10
SUT_CHAT_TIMEOUT_SECONDS = 600
JUDGE_GENERATE_TIMEOUT_SECONDS = 600


@dataclass(frozen=True)
class ClaimLevelTrialSummary:
    trial_number: int
    passed: bool
    result: ClaimLevelJudgeResult | None
    invalid: bool
    error: str | None


def format_verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def _count_status(result: ClaimLevelJudgeResult, status: str) -> int:
    return sum(1 for claim in result.claims if claim.support_status == status)


def _count_coverage(result: ClaimLevelJudgeResult, status: str) -> int:
    return sum(1 for item in result.question_coverage if item.status == status)


def _claims_mentioning(result: ClaimLevelJudgeResult, needle: str) -> list[str]:
    matches: list[str] = []
    lowered = needle.lower()
    for claim in result.claims:
        haystack = f"{claim.claim} {claim.candidate_excerpt}".lower()
        if lowered in haystack:
            matches.append(
                f"{claim.support_status} provenance={claim.provenance_issue}: {claim.claim}"
            )
    return matches


def run_sut_once(client: AnythingLLMClient, sut_prompt: str) -> str:
    workspace: PreparedWorkspace | None = None
    try:
        workspace = prepare_workspace(client, [AUTHORITATIVE_DOCUMENT_PATH])
        response = client.chat(
            workspace_slug=workspace.slug,
            message=sut_prompt,
            timeout=SUT_CHAT_TIMEOUT_SECONDS,
        )
        final_answer = response.final_answer
        if final_answer.strip() == "":
            msg = f"case {CASE_ID!r} produced an empty final answer"
            raise RuntimeError(msg)
        return final_answer
    finally:
        if workspace is not None:
            cleanup_workspace(client, workspace)


def run_judge_trial(
    *,
    ollama: OllamaClient,
    debug_run_dir: Path,
    trial_number: int,
    sut_prompt: str,
    prompt_sha256: str,
    candidate_answer: str,
    authoritative_source: str,
    reference_facts: list[str],
) -> ClaimLevelTrialSummary:
    judge_prompt = build_claim_level_judge_prompt(
        authoritative_source=authoritative_source,
        reference_facts=reference_facts,
        original_user_prompt=sut_prompt,
        candidate_answer=candidate_answer,
    )
    try:
        generation = ollama.generate_with_metadata(
            model=CLAIM_LEVEL_JUDGE_MODEL,
            prompt=judge_prompt,
            timeout=JUDGE_GENERATE_TIMEOUT_SECONDS,
            json_schema=build_claim_level_json_schema(),
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
            candidate_sha256=sha256_text(candidate_answer),
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
        return ClaimLevelTrialSummary(
            trial_number=trial_number,
            passed=claim_level_passed(result),
            result=result,
            invalid=False,
            error=None,
        )
    except RuntimeError as exc:
        return ClaimLevelTrialSummary(
            trial_number=trial_number,
            passed=False,
            result=None,
            invalid=True,
            error=str(exc),
        )


def print_trial_summary(trial: ClaimLevelTrialSummary) -> None:
    print()
    print(f"=== Judge trial {trial.trial_number} ===")
    if trial.invalid or trial.result is None:
        print("INVALID / incomplete claim audit")
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

    for label in ("Severity 0", "Severity 2.5", "Severity 5"):
        matches = _claims_mentioning(result, label)
        print(f"Claims involving {label}: {len(matches)}")
        for match in matches:
            print(f"  - {match}")

    derived = [
        claim for claim in result.claims if claim.support_status == "derived_from_unsupported_claim"
    ]
    print(f"derived_from_unsupported_claim findings: {len(derived)}")
    for claim in derived:
        print(f"  - {claim.claim}")
        print(f"    excerpt: {claim.candidate_excerpt}")
        print(f"    explanation: {claim.explanation}")

    if not trial.passed:
        print("Failure reasons:")
        for reason in claim_level_failure_reasons(result):
            print(f"  - {reason}")

    print("Rationale:")
    print(result.rationale)


def print_overall_summary(trials: list[ClaimLevelTrialSummary]) -> None:
    valid = [trial for trial in trials if not trial.invalid]
    invalid_count = len(trials) - len(valid)
    passed_count = sum(1 for trial in valid if trial.passed)
    failed_count = len(valid) - passed_count
    if valid:
        dominant_pass = passed_count >= failed_count
        dominant_count = max(passed_count, failed_count)
        stability = dominant_count / len(valid)
    else:
        dominant_pass = False
        stability = 0.0

    print()
    print("=== Claim-level stability summary ===")
    print()
    print(f"Valid judge trials: {len(valid)}/{len(trials)}")
    print(f"Invalid/incomplete audits: {invalid_count}")
    print(f"PASS: {passed_count}")
    print(f"FAIL: {failed_count}")
    print(f"Verdict stability: {stability * 100.0:.1f}%")
    print(f"Dominant verdict: {format_verdict(dominant_pass)}")


def main() -> int:
    prompt_sha256 = assert_prompt_fixture_integrity()
    sut_prompt = load_sut_prompt()
    authoritative_source = AUTHORITATIVE_DOCUMENT_PATH.read_text(encoding="utf-8")
    reference_facts = reference_facts_as_lines()
    debug_run_dir = create_judge_debug_run_dir(REPORTS_ROOT)

    print("Prompt fixture:")
    print(PROMPT_FIXTURE_RELATIVE_PATH)
    print()
    print("Prompt SHA-256:")
    print(prompt_sha256)
    print()
    print(
        "SUT prompt loaded without normalization "
        f"(length={len(sut_prompt)}, trailing_newline={sut_prompt.endswith(chr(10))})."
    )
    print("Exact fixture text is sent to the SUT as the single user message.")
    print()
    print(f"Judge debug traces:\n{debug_run_dir}")
    print()

    anythingllm = AnythingLLMClient()
    ollama = OllamaClient()

    print("Running SUT once...")
    candidate_answer = run_sut_once(anythingllm, sut_prompt)
    print()
    print("=== Fixed candidate answer ===")
    print(candidate_answer)
    print()

    trials: list[ClaimLevelTrialSummary] = []
    for trial_number in range(1, JUDGE_TRIALS + 1):
        print(f"Running claim-level judge trial {trial_number}/{JUDGE_TRIALS}...")
        trial = run_judge_trial(
            ollama=ollama,
            debug_run_dir=debug_run_dir,
            trial_number=trial_number,
            sut_prompt=sut_prompt,
            prompt_sha256=prompt_sha256,
            candidate_answer=candidate_answer,
            authoritative_source=authoritative_source,
            reference_facts=reference_facts,
        )
        trials.append(trial)
        print_trial_summary(trial)

    print_overall_summary(trials)
    print()
    print(f"Judge debug traces:\n{debug_run_dir / f'{CASE_ID}.jsonl'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
