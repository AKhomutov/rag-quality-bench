from dataclasses import dataclass


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
