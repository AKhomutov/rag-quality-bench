from judge_results import JudgeScore
from judge_stability import make_trial_result, summarize_judge_trials


def _score(
    groundedness: int,
    source_authority: int,
    unsupported_reasoning: int,
    answer_correctness: int,
) -> JudgeScore:
    return JudgeScore(
        groundedness=groundedness,
        source_authority=source_authority,
        unsupported_reasoning=unsupported_reasoning,
        answer_correctness=answer_correctness,
        rationale="test",
    )


def test_all_fail_stability_is_one() -> None:
    trials = [
        make_trial_result(
            trial_number=index,
            score=_score(0, 0, 0, 0),
            threshold=7,
            raw_json="{}",
        )
        for index in range(1, 11)
    ]
    summary = summarize_judge_trials(
        case_id="all_fail",
        trial_results=trials,
        deterministic_passed=False,
    )
    assert summary.passed_trials == 0
    assert summary.failed_trials == 10
    assert summary.pass_rate == 0.0
    assert summary.verdict_stability == 1.0
    assert summary.dominant_verdict_passed is False
    assert summary.deterministic_agreement_count == 10
    assert summary.deterministic_agreement_rate == 1.0
    assert summary.min_total == 0
    assert summary.max_total == 0
    assert summary.average_total == 0.0


def test_eight_fail_two_pass_stability_is_point_eight() -> None:
    trials = [
        make_trial_result(
            trial_number=1,
            score=_score(2, 2, 2, 2),
            threshold=7,
            raw_json="{}",
        ),
        make_trial_result(
            trial_number=2,
            score=_score(2, 2, 2, 1),
            threshold=7,
            raw_json="{}",
        ),
    ]
    trials.extend(
        make_trial_result(
            trial_number=index,
            score=_score(0, 0, 0, 0),
            threshold=7,
            raw_json="{}",
        )
        for index in range(3, 11)
    )
    summary = summarize_judge_trials(
        case_id="mixed",
        trial_results=trials,
        deterministic_passed=False,
    )
    assert summary.passed_trials == 2
    assert summary.failed_trials == 8
    assert summary.pass_rate == 0.2
    assert summary.verdict_stability == 0.8
    assert summary.dominant_verdict_passed is False
    assert summary.deterministic_agreement_count == 8
    assert summary.deterministic_agreement_rate == 0.8
    assert summary.min_total == 0
    assert summary.max_total == 8
    assert summary.average_total == 1.5


def test_per_dimension_averages() -> None:
    trials = [
        make_trial_result(
            trial_number=1,
            score=_score(0, 0, 1, 2),
            threshold=7,
            raw_json="{}",
        ),
        make_trial_result(
            trial_number=2,
            score=_score(2, 0, 1, 0),
            threshold=7,
            raw_json="{}",
        ),
    ]
    summary = summarize_judge_trials(
        case_id="dims",
        trial_results=trials,
        deterministic_passed=True,
    )
    by_name = {item.name: item for item in summary.dimensions}
    assert by_name["groundedness"].minimum == 0
    assert by_name["groundedness"].maximum == 2
    assert by_name["groundedness"].average == 1.0
    assert by_name["source_authority"].minimum == 0
    assert by_name["source_authority"].maximum == 0
    assert by_name["source_authority"].average == 0.0
    assert by_name["unsupported_reasoning"].average == 1.0
    assert by_name["answer_correctness"].minimum == 0
    assert by_name["answer_correctness"].maximum == 2
    assert by_name["answer_correctness"].average == 1.0
