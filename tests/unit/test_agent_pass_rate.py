from __future__ import annotations

from dataclasses import dataclass

from evaluation_results import EvaluationResult, TrialSummary


@dataclass(frozen=True)
class _FakeTrial:
    passed: bool
    failures: list[str]
    tools_used: list[str]


def _aggregate_pass_rate(
    trials: list[_FakeTrial],
    *,
    minimum_pass_rate: float,
) -> tuple[bool, float, list[str]]:
    """Mirror agent evaluator aggregation without live AnythingLLM calls."""
    total = len(trials)
    passed = sum(1 for trial in trials if trial.passed)
    pass_rate = passed / total if total else 0.0
    case_passed = pass_rate >= minimum_pass_rate
    failures: list[str] = []
    if not case_passed:
        failures.append(
            f"pass rate {pass_rate:.2f} is below required minimum {minimum_pass_rate:.2f}"
        )
        failures.append(f"{total - passed}/{total} trials failed")
    return case_passed, pass_rate, failures


def test_pass_rate_meets_threshold() -> None:
    trials = [_FakeTrial(True, [], ["rag-memory"]) for _ in range(9)]
    trials.append(_FakeTrial(False, ["forbidden"], ["web-browsing"]))
    passed, rate, failures = _aggregate_pass_rate(trials, minimum_pass_rate=0.9)
    assert passed is True
    assert rate == 0.9
    assert failures == []


def test_pass_rate_below_threshold_fails() -> None:
    trials = [
        _FakeTrial(True, [], ["rag-memory"]),
        _FakeTrial(False, ["missing tool"], []),
    ]
    passed, rate, failures = _aggregate_pass_rate(trials, minimum_pass_rate=0.9)
    assert passed is False
    assert rate == 0.5
    assert failures[0].startswith("pass rate 0.50")


def test_trial_summary_dto_fields() -> None:
    summary = TrialSummary(
        total=10,
        passed=9,
        failed=1,
        pass_rate=0.9,
        minimum_pass_rate=0.9,
        tool_usage={"rag-memory": 9},
    )
    result = EvaluationResult(
        suite="agent_tool_selection",
        case_id="prefer_workspace_rag_over_web",
        category="tool_selection",
        passed=True,
        expected_behavior="prefer rag",
        failure_description="used web",
        final_answer="Repeated agent evaluation: 9/10 trials passed.",
        failures=[],
        tools_used=["rag-memory"],
        trial_summary=summary,
    )
    assert result.passed is True
    assert result.trial_summary is not None
    assert result.trial_summary.pass_rate == 0.9
