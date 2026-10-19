from __future__ import annotations

from dataclasses import dataclass

from judge_results import JudgeScore, judge_passed, total_score


@dataclass(frozen=True)
class JudgeTrialResult:
    trial_number: int
    score: JudgeScore
    passed: bool
    raw_json: str


@dataclass(frozen=True)
class DimensionStats:
    name: str
    minimum: int
    maximum: int
    average: float


@dataclass(frozen=True)
class JudgeStabilitySummary:
    case_id: str
    trials: int
    passed_trials: int
    failed_trials: int
    pass_rate: float
    min_total: int
    max_total: int
    average_total: float
    verdict_stability: float
    dominant_verdict_passed: bool
    fact_check_inconsistency_count: int
    dimensions: tuple[DimensionStats, ...]


def _average(values: list[int]) -> float:
    if not values:
        msg = "cannot average empty values"
        raise ValueError(msg)
    return sum(values) / len(values)


def _dimension_stats(name: str, values: list[int]) -> DimensionStats:
    return DimensionStats(
        name=name,
        minimum=min(values),
        maximum=max(values),
        average=_average(values),
    )


def _trial_has_fact_check_inconsistency(score: JudgeScore) -> bool:
    """Flag unsupported_reasoning=2 while a fact_check is contradicted/misattributed."""
    if score.unsupported_reasoning.score != 2:
        return False
    for check in score.fact_checks:
        if check.assessment in ("contradicted", "misattributed"):
            return True
    return False


def summarize_judge_trials(
    case_id: str,
    trial_results: list[JudgeTrialResult],
) -> JudgeStabilitySummary:
    if not trial_results:
        msg = "trial_results must be non-empty"
        raise ValueError(msg)

    totals = [total_score(trial.score) for trial in trial_results]
    passed_trials = sum(1 for trial in trial_results if trial.passed)
    failed_trials = len(trial_results) - passed_trials
    trials = len(trial_results)
    dominant_count = max(passed_trials, failed_trials)
    dominant_verdict_passed = passed_trials >= failed_trials
    fact_check_inconsistency_count = sum(
        1 for trial in trial_results if _trial_has_fact_check_inconsistency(trial.score)
    )

    groundedness = [trial.score.groundedness.score for trial in trial_results]
    source_authority = [trial.score.source_authority.score for trial in trial_results]
    unsupported_reasoning = [trial.score.unsupported_reasoning.score for trial in trial_results]
    answer_correctness = [trial.score.answer_correctness.score for trial in trial_results]

    return JudgeStabilitySummary(
        case_id=case_id,
        trials=trials,
        passed_trials=passed_trials,
        failed_trials=failed_trials,
        pass_rate=passed_trials / trials,
        min_total=min(totals),
        max_total=max(totals),
        average_total=_average(totals),
        verdict_stability=dominant_count / trials,
        dominant_verdict_passed=dominant_verdict_passed,
        fact_check_inconsistency_count=fact_check_inconsistency_count,
        dimensions=(
            _dimension_stats("groundedness", groundedness),
            _dimension_stats("source_authority", source_authority),
            _dimension_stats("unsupported_reasoning", unsupported_reasoning),
            _dimension_stats("answer_correctness", answer_correctness),
        ),
    )


def make_trial_result(
    trial_number: int,
    score: JudgeScore,
    threshold: int,
    raw_json: str,
) -> JudgeTrialResult:
    return JudgeTrialResult(
        trial_number=trial_number,
        score=score,
        passed=judge_passed(score, threshold),
        raw_json=raw_json,
    )
