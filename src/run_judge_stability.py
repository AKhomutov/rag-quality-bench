from __future__ import annotations

from pathlib import Path

from adversarial_cases import AdversarialCase, load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient
from clients.ollama_client import JUDGE_MODEL, JUDGE_TEMPERATURE, OllamaClient
from conflicting_cases import ConflictingEvidenceCase, load_conflicting_evidence_cases
from evaluators.adversarial_evaluator import (
    run_adversarial_case,
    score_adversarial_answer,
)
from evaluators.conflicting_evidence_evaluator import (
    resolve_case_documents,
    run_conflicting_evidence_case,
    score_conflicting_evidence_answer,
)
from judge_debug import (
    case_trace_path,
    create_judge_debug_run_dir,
    parse_and_trace_judge_response,
)
from judge_results import JudgeScore, total_score
from judge_stability import (
    JudgeStabilitySummary,
    JudgeTrialResult,
    make_trial_result,
    summarize_judge_trials,
)
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

JUDGE_TRIALS = 10


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


def format_verdict(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def run_judge_trials(
    ollama: OllamaClient,
    *,
    debug_run_dir: Path,
    case_id: str,
    user_messages: list[str],
    authoritative_sources: list[str],
    reference_facts: list[str],
    candidate_answer: str,
    case_instruction: str | None,
    threshold: int,
    trials: int,
) -> list[JudgeTrialResult]:
    prompt = build_semantic_judge_prompt(
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        reference_facts=reference_facts,
        candidate_answer=candidate_answer,
        case_instruction=case_instruction,
    )
    trace_path = case_trace_path(debug_run_dir, case_id)
    results: list[JudgeTrialResult] = []
    for trial_number in range(1, trials + 1):
        raw_json = ollama.generate(model=JUDGE_MODEL, prompt=prompt)
        score = parse_and_trace_judge_response(
            path=trace_path,
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
        results.append(
            make_trial_result(
                trial_number=trial_number,
                score=score,
                threshold=threshold,
                raw_json=raw_json,
            )
        )
    return results


def print_score_diagnostic(score: JudgeScore) -> None:
    print("Fact checks:")
    for check in score.fact_checks:
        print(f'- "{check.fact}"')
        print(f"  candidate: {check.candidate_position}")
        print(f"  assessment: {check.assessment}")
    print()
    print("Scores:")
    print(f"groundedness: {score.groundedness.score}/2")
    print(f"  evidence: {score.groundedness.evidence}")
    print(f"source_authority: {score.source_authority.score}/2")
    print(f"  evidence: {score.source_authority.evidence}")
    print(f"unsupported_reasoning: {score.unsupported_reasoning.score}/2")
    print(f"  evidence: {score.unsupported_reasoning.evidence}")
    print(f"answer_correctness: {score.answer_correctness.score}/2")
    print(f"  evidence: {score.answer_correctness.evidence}")
    print()
    print("Rationale:")
    print(score.rationale)


def print_disagreement_or_flip(
    *,
    label: str,
    case_id: str,
    trial: JudgeTrialResult,
    deterministic_passed: bool,
) -> None:
    print()
    print(f"=== {label}: {case_id} trial {trial.trial_number} ===")
    print()
    print(f"Deterministic: {format_verdict(deterministic_passed)}")
    print(f"Judge: {format_verdict(trial.passed)} {total_score(trial.score)}/8")
    print()
    print_score_diagnostic(trial.score)


def print_case_stability(
    candidate_answer: str,
    deterministic_passed: bool,
    trial_results: list[JudgeTrialResult],
    summary: JudgeStabilitySummary,
) -> None:
    print()
    print(f"=== {summary.case_id} ===")
    print()
    print("Fixed candidate answer:")
    print(candidate_answer)
    print()
    print("Deterministic verdict:")
    print(format_verdict(deterministic_passed))
    print()
    print("Judge trials:")
    for trial in trial_results:
        verdict = format_verdict(trial.passed)
        print(f"{trial.trial_number}: {verdict} {total_score(trial.score)}/8")
    print()
    print("Summary:")
    print(f"PASS: {summary.passed_trials}/{summary.trials}")
    print(f"FAIL: {summary.failed_trials}/{summary.trials}")
    print(f"Pass rate: {summary.pass_rate * 100.0:.1f}%")
    print(f"Verdict stability: {summary.verdict_stability * 100.0:.1f}%")
    print(f"Dominant verdict: {format_verdict(summary.dominant_verdict_passed)}")
    print()
    print("Total score:")
    print(f"min: {summary.min_total}")
    print(f"max: {summary.max_total}")
    print(f"avg: {summary.average_total:.1f}")
    print()
    print("Dimensions:")
    for dimension in summary.dimensions:
        print(
            f"{dimension.name}: "
            f"min={dimension.minimum} "
            f"max={dimension.maximum} "
            f"avg={dimension.average:.1f}"
        )
    print()
    print("Agreement with deterministic:")
    print(f"{summary.deterministic_agreement_count}/{summary.trials}")
    print(f"{summary.deterministic_agreement_rate * 100.0:.1f}%")

    for trial in trial_results:
        is_disagreement = trial.passed != deterministic_passed
        is_flip = trial.passed != summary.dominant_verdict_passed
        if is_disagreement:
            print_disagreement_or_flip(
                label="DISAGREEMENT",
                case_id=summary.case_id,
                trial=trial,
                deterministic_passed=deterministic_passed,
            )
        elif is_flip:
            print_disagreement_or_flip(
                label="FLIP",
                case_id=summary.case_id,
                trial=trial,
                deterministic_passed=deterministic_passed,
            )


def print_overall_summary(summaries: list[JudgeStabilitySummary]) -> None:
    total_judge_calls = sum(item.trials for item in summaries)
    verdict_flips = sum(
        item.trials - max(item.passed_trials, item.failed_trials) for item in summaries
    )
    disagreements = sum(item.trials - item.deterministic_agreement_count for item in summaries)

    print()
    print("=== Judge stability summary ===")
    print()
    for summary in summaries:
        print(summary.case_id)
        print(f"dominant verdict: {format_verdict(summary.dominant_verdict_passed)}")
        print(f"verdict stability: {summary.verdict_stability * 100.0:.1f}%")
        print(f"agreement with deterministic: {summary.deterministic_agreement_rate * 100.0:.1f}%")
        print()

    print("Overall:")
    print(f"{total_judge_calls} judge calls")
    print(f"{verdict_flips} verdict flips relative to dominant result")
    print(f"{disagreements} disagreements with deterministic")

    unstable = [item for item in summaries if item.verdict_stability < 1.0]
    if unstable:
        print()
        print("Unstable cases:")
        for item in unstable:
            print(
                f"- {item.case_id}: "
                f"PASS {item.passed_trials}/{item.trials}, "
                f"FAIL {item.failed_trials}/{item.trials}"
            )


def _evaluate_case(
    spec: SemanticJudgeSpec,
    candidate_answer: str,
    user_messages: list[str],
    authoritative_sources: list[str],
    deterministic_passed: bool,
    ollama: OllamaClient,
    debug_run_dir: Path,
) -> JudgeStabilitySummary:
    trial_results = run_judge_trials(
        ollama=ollama,
        debug_run_dir=debug_run_dir,
        case_id=spec.case_id,
        user_messages=user_messages,
        authoritative_sources=authoritative_sources,
        reference_facts=spec.reference_facts,
        candidate_answer=candidate_answer,
        case_instruction=spec.case_instruction,
        threshold=spec.threshold,
        trials=JUDGE_TRIALS,
    )
    summary = summarize_judge_trials(
        case_id=spec.case_id,
        trial_results=trial_results,
        deterministic_passed=deterministic_passed,
    )
    print_case_stability(
        candidate_answer=candidate_answer,
        deterministic_passed=deterministic_passed,
        trial_results=trial_results,
        summary=summary,
    )
    return summary


def main() -> int:
    specs = load_semantic_judge_specs(SPECS_PATH)
    adversarial_by_id = _index_adversarial_cases()
    conflicting_by_id = _index_conflicting_cases()
    debug_run_dir = create_judge_debug_run_dir(REPORTS_ROOT)
    print(f"Judge debug traces:\n{debug_run_dir}")

    anythingllm = AnythingLLMClient()
    ollama = OllamaClient()
    support_sources = _read_support_policy()
    workspace: PreparedWorkspace | None = None
    summaries: list[JudgeStabilitySummary] = []

    try:
        workspace = prepare_workspace(anythingllm, STANDARD_DOCUMENTS)

        for spec in specs:
            if spec.source_suite == "adversarial":
                adversarial_case = adversarial_by_id.get(spec.case_id)
                if adversarial_case is None:
                    msg = f"adversarial case {spec.case_id!r} not found"
                    raise ValueError(msg)
                candidate_answer = run_adversarial_case(
                    anythingllm,
                    workspace.slug,
                    adversarial_case,
                )
                deterministic = score_adversarial_answer(
                    adversarial_case,
                    candidate_answer,
                )
                summaries.append(
                    _evaluate_case(
                        spec=spec,
                        candidate_answer=candidate_answer,
                        user_messages=[step.message for step in adversarial_case.steps],
                        authoritative_sources=support_sources,
                        deterministic_passed=deterministic.passed,
                        ollama=ollama,
                        debug_run_dir=debug_run_dir,
                    )
                )
            elif spec.source_suite == "conflicting_evidence":
                conflicting_case = conflicting_by_id.get(spec.case_id)
                if conflicting_case is None:
                    msg = f"conflicting case {spec.case_id!r} not found"
                    raise ValueError(msg)
                candidate_answer = run_conflicting_evidence_case(
                    anythingllm,
                    conflicting_case,
                )
                deterministic = score_conflicting_evidence_answer(
                    conflicting_case,
                    candidate_answer,
                )
                summaries.append(
                    _evaluate_case(
                        spec=spec,
                        candidate_answer=candidate_answer,
                        user_messages=[step.message for step in conflicting_case.steps],
                        authoritative_sources=_read_conflict_sources(conflicting_case),
                        deterministic_passed=deterministic.passed,
                        ollama=ollama,
                        debug_run_dir=debug_run_dir,
                    )
                )
            else:
                msg = f"unsupported source_suite {spec.source_suite!r}"
                raise ValueError(msg)
    finally:
        if workspace is not None:
            cleanup_workspace(anythingllm, workspace)

    print_overall_summary(summaries)
    print()
    print(f"Judge debug traces:\n{debug_run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
