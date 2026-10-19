from __future__ import annotations

from clients.anythingllm_client import AnythingLLMClient
from clients.ollama_client import JUDGE_MODEL, OllamaClient
from conflicting_cases import ConflictingEvidenceCase, load_conflicting_evidence_cases
from evaluators.conflicting_evidence_evaluator import (
    resolve_case_documents,
    run_conflicting_evidence_case,
    score_conflicting_evidence_answer,
)
from judge_results import (
    JUDGE_PASS_THRESHOLD,
    JudgeScore,
    judge_passed,
    parse_judge_score,
    total_score,
)
from judges.conflict_judge import build_conflict_judge_prompt
from workspace_setup import PROJECT_ROOT

CASE_ID = "conflicting_refund_policies"
CONFLICTING_CASES_PATH = PROJECT_ROOT / "fixtures" / "datasets" / "conflicting_evidence.yaml"

CLEAR_PASS_ANSWER = (
    "The two authoritative policies conflict. One allows refunds for monthly "
    "subscriptions within 7 calendar days of the initial purchase, while the "
    "other says all subscription payments are final and non-refundable with no "
    "exceptions. The supplied sources do not define which policy takes "
    "precedence, so refund eligibility cannot be determined from them alone."
)

CLEAR_FAIL_ANSWER = "Yes. The 7-day refund rule is the only exception to the no-refund policy."

MIXED_ANSWER = (
    "The documents conflict, but the 7-day rule is more specific, so it should take precedence."
)


def load_target_case() -> ConflictingEvidenceCase:
    cases = load_conflicting_evidence_cases(CONFLICTING_CASES_PATH)
    for case in cases:
        if case.id == CASE_ID:
            return case
    msg = f"case {CASE_ID!r} not found in {CONFLICTING_CASES_PATH}"
    raise ValueError(msg)


def load_authoritative_sources(case: ConflictingEvidenceCase) -> list[str]:
    sources: list[str] = []
    for document_path in resolve_case_documents(case):
        sources.append(document_path.read_text(encoding="utf-8"))
    return sources


def score_with_judge(
    ollama: OllamaClient,
    user_messages: list[str],
    authoritative_sources: list[str],
    candidate_answer: str,
) -> tuple[str, str, JudgeScore]:
    prompt = build_conflict_judge_prompt(
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        candidate_answer=candidate_answer,
    )
    raw_json = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
    score = parse_judge_score(raw_json)
    return prompt, raw_json, score


def format_verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_judge_block(score: JudgeScore) -> None:
    print("Judge scores:")
    print(f"  groundedness: {score.groundedness}/2")
    print(f"  conflict_handling: {score.conflict_handling}/2")
    print(f"  unsupported_precedence: {score.unsupported_precedence}/2")
    print(f"  answer_correctness: {score.answer_correctness}/2")
    print()
    print(f"Total: {total_score(score)}/8")
    print(f"Threshold: {JUDGE_PASS_THRESHOLD}/8")
    print(f"Judge verdict: {format_verdict(judge_passed(score))}")
    print()
    print("Rationale:")
    print(score.rationale)


def print_probe_section(
    title: str,
    ollama: OllamaClient,
    user_messages: list[str],
    authoritative_sources: list[str],
    candidate_answer: str,
) -> None:
    print()
    print("=" * 72)
    print(f"Probe: {title}")
    print("=" * 72)
    print()
    print("Candidate answer:")
    print(candidate_answer)
    print()
    _prompt, raw_json, score = score_with_judge(
        ollama=ollama,
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        candidate_answer=candidate_answer,
    )
    print("Judge raw JSON:")
    print(raw_json)
    print()
    print_judge_block(score)


def main() -> int:
    case = load_target_case()
    user_messages = [step.message for step in case.steps]
    authoritative_sources = load_authoritative_sources(case)

    print(f"Case: {case.id}")
    print()
    print("Running SUT conversation once...")
    anythingllm = AnythingLLMClient()
    candidate_answer = run_conflicting_evidence_case(anythingllm, case)

    deterministic = score_conflicting_evidence_answer(case, candidate_answer)
    ollama = OllamaClient()
    prompt, raw_json, judge_score = score_with_judge(
        ollama=ollama,
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        candidate_answer=candidate_answer,
    )

    print()
    print("Final judge prompt:")
    print(prompt)
    print()
    print("Candidate answer:")
    print(candidate_answer)
    print()
    print("Deterministic oracle:")
    print(f"  {format_verdict(deterministic.passed)}")
    if deterministic.failures:
        for failure in deterministic.failures:
            print(f"  - {failure}")
    print()
    print("Judge raw JSON:")
    print(raw_json)
    print()
    print("Parsed JudgeScore:")
    print(judge_score)
    print()
    print_judge_block(judge_score)
    print()
    agreement = "YES" if deterministic.passed == judge_passed(judge_score) else "NO"
    print(f"Oracle agreement: {agreement}")

    print_probe_section(
        "clear PASS",
        ollama,
        user_messages,
        authoritative_sources,
        CLEAR_PASS_ANSWER,
    )
    print_probe_section(
        "clear FAIL",
        ollama,
        user_messages,
        authoritative_sources,
        CLEAR_FAIL_ANSWER,
    )
    print_probe_section(
        "mixed",
        ollama,
        user_messages,
        authoritative_sources,
        MIXED_ANSWER,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
