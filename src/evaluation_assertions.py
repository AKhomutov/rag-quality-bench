from evaluation_results import EvaluationResult


def format_evaluation_failure(result: EvaluationResult) -> str:
    failure_lines: list[str] = []
    for failure in result.failures:
        failure_lines.append(f"- {failure}")
    failures_text = "\n".join(failure_lines)

    return (
        f"suite: {result.suite}\n"
        f"case: {result.case_id}\n"
        f"category: {result.category}\n"
        f"expected behavior: {result.expected_behavior}\n"
        f"failure: {result.failure_description}\n"
        f"observed failures:\n{failures_text}\n"
        f"final answer:\n{result.final_answer}"
    )
