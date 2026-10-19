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
    deterministic_agreement_count: int
    deterministic_agreement_rate: float
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


def summarize_judge_trials(
    case_id: str,
    trial_results: list[JudgeTrialResult],
    deterministic_passed: bool,
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
    agreement_count = sum(1 for trial in trial_results if trial.passed == deterministic_passed)

    groundedness = [trial.score.groundedness for trial in trial_results]
    source_authority = [trial.score.source_authority for trial in trial_results]
    unsupported_reasoning = [trial.score.unsupported_reasoning for trial in trial_results]
    answer_correctness = [trial.score.answer_correctness for trial in trial_results]

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
        deterministic_agreement_count=agreement_count,
        deterministic_agreement_rate=agreement_count / trials,
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
