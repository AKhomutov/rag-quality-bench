from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from adversarial_cases import AdversarialCase, load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient
from clients.ollama_client import JUDGE_MODEL, JUDGE_TEMPERATURE, OllamaClient
from conflicting_cases import ConflictingEvidenceCase, load_conflicting_evidence_cases
from evaluators.adversarial_evaluator import run_adversarial_case
from evaluators.conflicting_evidence_evaluator import (
    resolve_case_documents,
    run_conflicting_evidence_case,
)
from judge_debug import (
    case_trace_path,
    create_judge_debug_run_dir,
    parse_and_trace_judge_response,
)
from judge_results import JudgeScore, judge_passed, total_score
from judges.semantic_judge import build_semantic_judge_prompt
from semantic_judge_specs import SemanticJudgeSpec, load_semantic_judge_specs
from workspace_setup import (
    PROJECT_ROOT,
    STANDARD_DOCUMENTS,
    PreparedWorkspace,
    cleanup_workspace,
    prepare_workspace,
)

SPECS_PATH = PROJECT_ROOT / "fixtures" / "datasets" / "semantic_judge_specs.yaml"
ADVERSARIAL_CASES_PATH = PROJECT_ROOT / "fixtures" / "datasets" / "adversarial_conversations.yaml"
CONFLICTING_CASES_PATH = PROJECT_ROOT / "fixtures" / "datasets" / "conflicting_evidence.yaml"
SUPPORT_POLICY_PATH = PROJECT_ROOT / "fixtures" / "documents" / "acme_support_policy.md"
REPORTS_ROOT = PROJECT_ROOT / "reports"

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


@dataclass(frozen=True)
class CaseJudgeResult:
    case_id: str
    judge_passed: bool
    total: int


def _index_adversarial_cases() -> dict[str, AdversarialCase]:
    return {case.id: case for case in load_adversarial_cases(ADVERSARIAL_CASES_PATH)}


def _index_conflicting_cases() -> dict[str, ConflictingEvidenceCase]:
    return {case.id: case for case in load_conflicting_evidence_cases(CONFLICTING_CASES_PATH)}


def _read_support_policy() -> list[str]:
    return [SUPPORT_POLICY_PATH.read_text(encoding="utf-8")]


def _read_conflict_sources(case: ConflictingEvidenceCase) -> list[str]:
    return [
        document_path.read_text(encoding="utf-8") for document_path in resolve_case_documents(case)
    ]


def score_with_judge(
    ollama: OllamaClient,
    *,
    debug_run_dir: Path,
    case_id: str,
    trial_number: int | None,
    user_messages: list[str],
    authoritative_sources: list[str],
    reference_facts: list[str],
    candidate_answer: str,
    case_instruction: str | None,
    threshold: int,
) -> tuple[str, str, JudgeScore]:
    prompt = build_semantic_judge_prompt(
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        reference_facts=reference_facts,
        candidate_answer=candidate_answer,
        case_instruction=case_instruction,
    )
    raw_json = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
    score = parse_and_trace_judge_response(
        path=case_trace_path(debug_run_dir, case_id),
        case_id=case_id,
        trial_number=trial_number,
        candidate_answer=candidate_answer,
        reference_facts=reference_facts,
        case_instruction=case_instruction,
        prompt=prompt,
        raw_response=raw_json,
        threshold=threshold,
        model=JUDGE_MODEL,
        temperature=float(JUDGE_TEMPERATURE),
    )
    return prompt, raw_json, score


def format_verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def print_fact_checks(score: JudgeScore) -> None:
    print("Fact checks:")
    for check in score.fact_checks:
        print(f'- "{check.fact}"')
        print(f"  candidate: {check.candidate_position}")
        print(f"  assessment: {check.assessment}")
    print()


def print_judge_block(score: JudgeScore, threshold: int) -> None:
    print_fact_checks(score)
    print("LLM judge scores:")
    print(f"  groundedness: {score.groundedness.score}/2")
    print(f"  source authority: {score.source_authority.score}/2")
    print(f"  unsupported reasoning: {score.unsupported_reasoning.score}/2")
    print(f"  answer correctness: {score.answer_correctness.score}/2")
    print()
    print(f"Total: {total_score(score)}/8")
    print(f"Threshold: {threshold}/8")
    print(f"Judge verdict: {format_verdict(judge_passed(score, threshold))}")
    print()
    print("Rationale:")
    print(score.rationale)


def print_case_result(
    case_id: str,
    candidate_answer: str,
    score: JudgeScore,
    threshold: int,
) -> CaseJudgeResult:
    print()
    print("=" * 72)
    print(f"Case: {case_id}")
    print("=" * 72)
    print()
    print("Candidate answer:")
    print(candidate_answer)
    print()
    print_judge_block(score, threshold)
    return CaseJudgeResult(
        case_id=case_id,
        judge_passed=judge_passed(score, threshold),
        total=total_score(score),
    )


def print_summary(results: list[CaseJudgeResult]) -> None:
    total = len(results)
    passed = sum(1 for item in results if item.judge_passed)
    failed = total - passed

    print()
    print("=" * 72)
    print("Semantic judge summary")
    print("=" * 72)
    print()
    print(f"Cases: {total}")
    print(f"Judge PASS: {passed}")
    print(f"Judge FAIL: {failed}")
    print()
    for item in results:
        print(f"- {item.case_id}: {format_verdict(item.judge_passed)} ({item.total}/8)")


def print_probe_section(
    title: str,
    ollama: OllamaClient,
    debug_run_dir: Path,
    probe_case_id: str,
    user_messages: list[str],
    authoritative_sources: list[str],
    reference_facts: list[str],
    candidate_answer: str,
    case_instruction: str | None,
    threshold: int,
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
        debug_run_dir=debug_run_dir,
        case_id=probe_case_id,
        trial_number=None,
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        reference_facts=reference_facts,
        candidate_answer=candidate_answer,
        case_instruction=case_instruction,
        threshold=threshold,
    )
    print("Judge raw JSON:")
    print(raw_json)
    print()
    print_judge_block(score, threshold)


def _run_adversarial_spec(
    spec: SemanticJudgeSpec,
    case: AdversarialCase,
    client: AnythingLLMClient,
    workspace_slug: str,
    ollama: OllamaClient,
    authoritative_sources: list[str],
    debug_run_dir: Path,
) -> CaseJudgeResult:
    user_messages = [step.message for step in case.steps]
    candidate_answer = run_adversarial_case(client, workspace_slug, case)
    _prompt, _raw, score = score_with_judge(
        ollama=ollama,
        debug_run_dir=debug_run_dir,
        case_id=spec.case_id,
        trial_number=None,
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        reference_facts=spec.reference_facts,
        candidate_answer=candidate_answer,
        case_instruction=spec.case_instruction,
        threshold=spec.threshold,
    )
    return print_case_result(
        case_id=spec.case_id,
        candidate_answer=candidate_answer,
        score=score,
        threshold=spec.threshold,
    )


def _run_conflict_spec(
    spec: SemanticJudgeSpec,
    case: ConflictingEvidenceCase,
    client: AnythingLLMClient,
    ollama: OllamaClient,
    debug_run_dir: Path,
) -> tuple[CaseJudgeResult, list[str], list[str]]:
    user_messages = [step.message for step in case.steps]
    authoritative_sources = _read_conflict_sources(case)
    candidate_answer = run_conflicting_evidence_case(client, case)
    prompt, _raw, score = score_with_judge(
        ollama=ollama,
        debug_run_dir=debug_run_dir,
        case_id=spec.case_id,
        trial_number=None,
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        reference_facts=spec.reference_facts,
        candidate_answer=candidate_answer,
        case_instruction=spec.case_instruction,
        threshold=spec.threshold,
    )
    print()
    print("Final generic rubric prompt (from conflict case):")
    print(prompt)
    result = print_case_result(
        case_id=spec.case_id,
        candidate_answer=candidate_answer,
        score=score,
        threshold=spec.threshold,
    )
    return result, user_messages, authoritative_sources


def main() -> int:
    specs = load_semantic_judge_specs(SPECS_PATH)
    adversarial_by_id = _index_adversarial_cases()
    conflicting_by_id = _index_conflicting_cases()
    debug_run_dir = create_judge_debug_run_dir(REPORTS_ROOT)
    print(f"Judge debug traces:\n{debug_run_dir}")

    print()
    print("Judge specs:")
    for spec in specs:
        instruction = "yes" if spec.case_instruction is not None else "no"
        print(
            f"- {spec.case_id} ({spec.source_suite}), "
            f"threshold={spec.threshold}, "
            f"facts={len(spec.reference_facts)}, "
            f"case_instruction={instruction}"
        )
        for index, fact in enumerate(spec.reference_facts, start=1):
            print(f"    {index}. {fact}")

    anythingllm = AnythingLLMClient()
    ollama = OllamaClient()
    support_sources = _read_support_policy()
    workspace: PreparedWorkspace | None = None
    results: list[CaseJudgeResult] = []
    probe_user_messages: list[str] | None = None
    probe_sources: list[str] | None = None
    probe_facts: list[str] | None = None
    probe_instruction: str | None = None
    probe_threshold = 7

    try:
        workspace = prepare_workspace(anythingllm, STANDARD_DOCUMENTS)

        for spec in specs:
            if spec.source_suite == "adversarial":
                adversarial_case = adversarial_by_id.get(spec.case_id)
                if adversarial_case is None:
                    msg = f"adversarial case {spec.case_id!r} not found"
                    raise ValueError(msg)
                results.append(
                    _run_adversarial_spec(
                        spec=spec,
                        case=adversarial_case,
                        client=anythingllm,
                        workspace_slug=workspace.slug,
                        ollama=ollama,
                        authoritative_sources=support_sources,
                        debug_run_dir=debug_run_dir,
                    )
                )
            elif spec.source_suite == "conflicting_evidence":
                conflicting_case = conflicting_by_id.get(spec.case_id)
                if conflicting_case is None:
                    msg = f"conflicting case {spec.case_id!r} not found"
                    raise ValueError(msg)
                result, probe_user_messages, probe_sources = _run_conflict_spec(
                    spec=spec,
                    case=conflicting_case,
                    client=anythingllm,
                    ollama=ollama,
                    debug_run_dir=debug_run_dir,
                )
                probe_facts = spec.reference_facts
                probe_instruction = spec.case_instruction
                probe_threshold = spec.threshold
                results.append(result)
            else:
                msg = f"unsupported source_suite {spec.source_suite!r}"
                raise ValueError(msg)
    finally:
        if workspace is not None:
            cleanup_workspace(anythingllm, workspace)

    print_summary(results)

    if probe_user_messages is None or probe_sources is None or probe_facts is None:
        msg = "conflict probe context was not prepared"
        raise RuntimeError(msg)

    print_probe_section(
        "clear PASS",
        ollama,
        debug_run_dir,
        "probe_clear_pass",
        probe_user_messages,
        probe_sources,
        probe_facts,
        CLEAR_PASS_ANSWER,
        probe_instruction,
        probe_threshold,
    )
    print_probe_section(
        "clear FAIL",
        ollama,
        debug_run_dir,
        "probe_clear_fail",
        probe_user_messages,
        probe_sources,
        probe_facts,
        CLEAR_FAIL_ANSWER,
        probe_instruction,
        probe_threshold,
    )
    print_probe_section(
        "mixed",
        ollama,
        debug_run_dir,
        "probe_mixed",
        probe_user_messages,
        probe_sources,
        probe_facts,
        MIXED_ANSWER,
        probe_instruction,
        probe_threshold,
    )

    print()
    print(f"Judge debug traces:\n{debug_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
