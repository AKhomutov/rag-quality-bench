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

    trial_section = ""
    if result.trial_summary is not None:
        summary = result.trial_summary
        pass_rate_pct = summary.pass_rate * 100.0
        required_pct = summary.minimum_pass_rate * 100.0
        trial_section = (
            f"trials: {summary.passed}/{summary.total} passed "
            f"({pass_rate_pct:.1f}%, required {required_pct:.1f}%)\n"
        )
        usage_lines: list[str] = []
        for tool_name, count in summary.tool_usage.items():
            usage_lines.append(f"- {tool_name}: {count}/{summary.total}")
        if usage_lines:
            usage_text = "\n".join(usage_lines)
            trial_section = f"{trial_section}tool selection by trial:\n{usage_text}\n"

    return (
        f"suite: {result.suite}\n"
        f"case: {result.case_id}\n"
        f"category: {result.category}\n"
        f"expected behavior: {result.expected_behavior}\n"
        f"failure: {result.failure_description}\n"
        f"{trial_section}"
        f"{tools_section}"
        f"observed failures:\n{failures_text}\n"
        f"final answer:\n{result.final_answer}"
    )
