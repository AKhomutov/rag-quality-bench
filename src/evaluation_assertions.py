from evaluation_results import EvaluationResult


def format_evaluation_failure(result: EvaluationResult) -> str:
    failure_lines: list[str] = []
    for failure in result.failures:
        failure_lines.append(f"- {failure}")
    failures_text = "\n".join(failure_lines)

    tools_section = ""
    if result.tools_used:
        tool_lines: list[str] = []
        for tool_name in result.tools_used:
            tool_lines.append(f"- {tool_name}")
        tools_text = "\n".join(tool_lines)
        tools_section = f"tools used:\n{tools_text}\n"

    return (
        f"suite: {result.suite}\n"
        f"case: {result.case_id}\n"
        f"category: {result.category}\n"
        f"expected behavior: {result.expected_behavior}\n"
        f"failure: {result.failure_description}\n"
        f"{tools_section}"
        f"observed failures:\n{failures_text}\n"
        f"final answer:\n{result.final_answer}"
    )
