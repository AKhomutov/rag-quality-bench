import json
from pathlib import Path

from adversarial_cases import load_adversarial_cases
from agent_cases import load_agent_tool_selection_cases
from clients.anythingllm_client import AnythingLLMClient
from conflicting_cases import load_conflicting_evidence_cases
from evaluation_results import EvaluationResult, TrialSummary
from evaluators.agent_tool_selection_evaluator import evaluate_agent_tool_selection_case
from evaluators.baseline_evaluator import evaluate_rag_case
from rag_cases import load_rag_cases
from workspace_setup import (
    STANDARD_DOCUMENTS,
    PreparedWorkspace,
    cleanup_workspace,
    prepare_workspace,
)

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CASES_PATH = ROOT / "fixtures" / "datasets" / "rag_cases.yaml"
ADVERSARIAL_CASES_PATH = ROOT / "fixtures" / "datasets" / "adversarial_conversations.yaml"
CONFLICTING_CASES_PATH = ROOT / "fixtures" / "datasets" / "conflicting_evidence.yaml"
AGENT_CASES_PATH = ROOT / "fixtures" / "datasets" / "agent_tool_selection.yaml"
REPORTS_DIR = ROOT / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "evaluation_report.json"
MARKDOWN_REPORT_PATH = REPORTS_DIR / "evaluation_report.md"


def count_passed(results: list[EvaluationResult]) -> int:
    passed_count = 0
    for result in results:
        if result.passed:
            passed_count += 1
    return passed_count


def count_suite_stats(results: list[EvaluationResult], suite: str) -> dict[str, int]:
    total = 0
    passed = 0
    for result in results:
        if result.suite != suite:
            continue
        total += 1
        if result.passed:
            passed += 1
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
    }


def count_category_stats(
    results: list[EvaluationResult],
    category: str,
) -> dict[str, int]:
    total = 0
    passed = 0
    for result in results:
        if result.category != category:
            continue
        total += 1
        if result.passed:
            passed += 1
    return {
        "total": total,
        "passed": passed,
        "failed": total - passed,
    }


def collect_categories(results: list[EvaluationResult]) -> list[str]:
    categories: list[str] = []
    for result in results:
        if result.category not in categories:
            categories.append(result.category)
    return categories


def collect_suites(results: list[EvaluationResult]) -> list[str]:
    suites: list[str] = []
    for result in results:
        if result.suite not in suites:
            suites.append(result.suite)
    return suites


def build_categories_payload(results: list[EvaluationResult]) -> dict[str, object]:
    categories_data: dict[str, object] = {}
    categories = collect_categories(results)
    for category in categories:
        categories_data[category] = count_category_stats(results, category)
    return categories_data


def build_suites_payload(results: list[EvaluationResult]) -> dict[str, object]:
    suites_data: dict[str, object] = {}
    suites = collect_suites(results)
    for suite in suites:
        suites_data[suite] = count_suite_stats(results, suite)
    return suites_data


def format_suite_label(suite: str) -> str:
    parts: list[str] = []
    for part in suite.split("_"):
        if not part:
            continue
        parts.append(part[:1].upper() + part[1:])
    return " ".join(parts)


def trial_summary_to_json_object(summary: TrialSummary | None) -> dict[str, object] | None:
    if summary is None:
        return None

    return {
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "pass_rate": summary.pass_rate,
        "minimum_pass_rate": summary.minimum_pass_rate,
        "tool_usage": summary.tool_usage,
    }


def result_to_json_object(result: EvaluationResult) -> dict[str, object]:
    return {
        "suite": result.suite,
        "case_id": result.case_id,
        "category": result.category,
        "passed": result.passed,
        "expected_behavior": result.expected_behavior,
        "failure_description": result.failure_description,
        "failures": result.failures,
        "final_answer": result.final_answer,
        "tools_used": result.tools_used,
        "trial_summary": trial_summary_to_json_object(result.trial_summary),
    }


def build_json_payload(results: list[EvaluationResult]) -> dict[str, object]:
    passed_count = count_passed(results)
    failed_count = len(results) - passed_count

    results_data: list[dict[str, object]] = []
    for result in results:
        results_data.append(result_to_json_object(result))

    return {
        "summary": {
            "total": len(results),
            "passed": passed_count,
            "failed": failed_count,
        },
        "suites": build_suites_payload(results),
        "categories": build_categories_payload(results),
        "results": results_data,
    }


def format_blockquote(text: str) -> str:
    lines = text.splitlines()
    if not lines:
        return ">"

    quoted_lines: list[str] = []
    for line in lines:
        quoted_lines.append(f"> {line}")
    return "\n".join(quoted_lines)


def append_result_section(sections: list[str], result: EvaluationResult) -> None:
    if result.passed:
        status = "PASS"
    else:
        status = "FAIL"

    sections.append(f"### {status} — {result.case_id}")
    sections.append("")
    sections.append(f"**Category:** {result.category}")
    sections.append("")
    sections.append(f"**Expected behavior:** {result.expected_behavior}")
    sections.append("")

    if result.trial_summary is not None:
        summary = result.trial_summary
        pass_rate_pct = summary.pass_rate * 100.0
        required_pct = summary.minimum_pass_rate * 100.0
        sections.append(f"**Trials:** {summary.total}")
        sections.append(f"**Passed trials:** {summary.passed}")
        sections.append(f"**Failed trials:** {summary.failed}")
        sections.append(f"**Pass rate:** {pass_rate_pct:.1f}%")
        sections.append(f"**Required pass rate:** {required_pct:.1f}%")
        sections.append("")
        sections.append("**Tool selection by trial:**")
        for tool_name, count in summary.tool_usage.items():
            sections.append(f"- {tool_name}: {count}/{summary.total}")
        sections.append("")
    elif result.tools_used:
        sections.append("**Tools used:**")
        for tool_name in result.tools_used:
            sections.append(f"- {tool_name}")
        sections.append("")

    if not result.passed:
        sections.append(f"**Failure classification:** {result.failure_description}")
        sections.append("")
        sections.append("**Observed failures:**")
        for failure in result.failures:
            sections.append(f"- {failure}")
        sections.append("")
        sections.append("**Final answer:**")
        sections.append("")
        sections.append(format_blockquote(result.final_answer))
        sections.append("")


def filter_results_by_suite(
    results: list[EvaluationResult],
    suite: str,
) -> list[EvaluationResult]:
    filtered: list[EvaluationResult] = []
    for result in results:
        if result.suite == suite:
            filtered.append(result)
    return filtered


def build_markdown_report(results: list[EvaluationResult]) -> str:
    overall_passed = count_passed(results)
    overall_failed = len(results) - overall_passed
    suites = collect_suites(results)

    sections: list[str] = [
        "# RAG Quality Evaluation Report",
        "",
        "## Summary",
        "",
        "| Suite | Total | Passed | Failed |",
        "| --- | ---: | ---: | ---: |",
    ]

    for suite in suites:
        suite_stats = count_suite_stats(results, suite)
        sections.append(
            f"| {format_suite_label(suite)} | {suite_stats['total']} | "
            f"{suite_stats['passed']} | {suite_stats['failed']} |"
        )

    sections.append(
        f"| **Overall** | **{len(results)}** | **{overall_passed}** | **{overall_failed}** |"
    )
    sections.append("")
    sections.append("## Category Breakdown")
    sections.append("")
    sections.append("| Category | Total | Passed | Failed |")
    sections.append("| --- | ---: | ---: | ---: |")

    categories = collect_categories(results)
    for category in categories:
        category_stats = count_category_stats(results, category)
        sections.append(
            f"| {category} | {category_stats['total']} | "
            f"{category_stats['passed']} | {category_stats['failed']} |"
        )

    for suite in suites:
        sections.append("")
        sections.append(f"## {format_suite_label(suite)} Results")
        sections.append("")
        suite_results = filter_results_by_suite(results, suite)
        for result in suite_results:
            append_result_section(sections, result)

    return "\n".join(sections)


def write_json_report(path: Path, results: list[EvaluationResult]) -> None:
    payload = build_json_payload(results)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_markdown_report(path: Path, results: list[EvaluationResult]) -> None:
    content = build_markdown_report(results)
    path.write_text(content, encoding="utf-8")


def print_status(result: EvaluationResult) -> None:
    if result.passed:
        status = "PASS"
    else:
        status = "FAIL"
    print(f"  {status}")

    if result.trial_summary is None:
        return

    summary = result.trial_summary
    pass_rate_pct = summary.pass_rate * 100.0
    required_pct = summary.minimum_pass_rate * 100.0
    print(
        f"  trials: {summary.passed}/{summary.total} passed "
        f"({pass_rate_pct:.1f}%, required {required_pct:.1f}%)"
    )
    print("  tools:")
    for tool_name, count in summary.tool_usage.items():
        print(f"    {tool_name}: {count}/{summary.total}")


def main() -> int:
    client = AnythingLLMClient()
    workspace: PreparedWorkspace | None = None
    exit_code = 1

    try:
        print("Preparing temporary workspace...")
        workspace = prepare_workspace(client, STANDARD_DOCUMENTS)
        print(f"Workspace ready: {workspace.slug}")
        print()

        results: list[EvaluationResult] = []

        print("Baseline evaluations")
        print("--------------------")
        baseline_cases = load_rag_cases(BASELINE_CASES_PATH)
        for baseline_case in baseline_cases:
            print(f"Evaluating {baseline_case.id}...")
            result = evaluate_rag_case(
                client=client,
                workspace_slug=workspace.slug,
                case=baseline_case,
            )
            print_status(result)
            results.append(result)

        print()
        print("Adversarial evaluations (semantic-only)")
        print("---------------------------------------")
        adversarial_cases = load_adversarial_cases(ADVERSARIAL_CASES_PATH)
        for adversarial_case in adversarial_cases:
            print(
                f"  {adversarial_case.id}: semantic-only "
                "(LLM judge via run_judge_prototype / run_judge_stability); "
                "excluded from deterministic totals"
            )

        print()
        print("Conflicting evidence evaluations (semantic-only)")
        print("------------------------------------------------")
        conflicting_cases = load_conflicting_evidence_cases(CONFLICTING_CASES_PATH)
        for conflicting_case in conflicting_cases:
            print(
                f"  {conflicting_case.id}: semantic-only "
                "(LLM judge via run_judge_prototype / run_judge_stability); "
                "excluded from deterministic totals"
            )

        print()
        print("Agent tool-selection evaluations")
        print("--------------------------------")
        agent_cases = load_agent_tool_selection_cases(AGENT_CASES_PATH)
        for agent_case in agent_cases:
            print(f"Evaluating {agent_case.id}...")
            result = evaluate_agent_tool_selection_case(
                client=client,
                case=agent_case,
            )
            print_status(result)
            results.append(result)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        write_json_report(JSON_REPORT_PATH, results)
        write_markdown_report(MARKDOWN_REPORT_PATH, results)

        passed_count = count_passed(results)
        failed_count = len(results) - passed_count

        print()
        suites = collect_suites(results)
        for suite in suites:
            suite_stats = count_suite_stats(results, suite)
            print(f"{suite}: {suite_stats['passed']} passed / {suite_stats['failed']} failed")
        print(f"overall: {passed_count} passed / {failed_count} failed")
        print("categories:")
        categories = collect_categories(results)
        for category in categories:
            category_stats = count_category_stats(results, category)
            print(
                f"  {category}: "
                f"{category_stats['passed']} passed / {category_stats['failed']} failed"
            )
        print(f"Wrote {JSON_REPORT_PATH}")
        print(f"Wrote {MARKDOWN_REPORT_PATH}")

        if failed_count > 0:
            exit_code = 1
        else:
            exit_code = 0
    finally:
        if workspace is not None:
            print()
            print(f"Cleaning up workspace: {workspace.slug}")
            cleanup_workspace(client, workspace)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
