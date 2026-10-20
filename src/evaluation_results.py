from dataclasses import dataclass


@dataclass(frozen=True)
class TrialSummary:
    total: int
    passed: int
    failed: int
    pass_rate: float
    minimum_pass_rate: float
    tool_usage: dict[str, int]


@dataclass(frozen=True)
class EvaluationResult:
    suite: str
    case_id: str
    category: str
    passed: bool
    expected_behavior: str
    failure_description: str
    final_answer: str
    failures: list[str]
    tools_used: list[str]
    trial_summary: TrialSummary | None
    # Optional fields for suites that distinguish ERROR from semantic FAIL.
    execution_error: str | None = None
    long_form_metrics: dict[str, object] | None = None
    debug_path: str | None = None
