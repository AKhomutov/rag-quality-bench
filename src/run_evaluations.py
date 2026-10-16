import json
from pathlib import Path

from adversarial_cases import load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient
from evaluation_results import EvaluationResult
from evaluators.adversarial_evaluator import evaluate_adversarial_case
from evaluators.baseline_evaluator import evaluate_rag_case
from rag_cases import load_rag_cases
from workspace_setup import STANDARD_DOCUMENTS, cleanup_workspace, prepare_workspace

ROOT = Path(__file__).resolve().parents[1]
BASELINE_CASES_PATH = ROOT / "fixtures" / "datasets" / "rag_cases.yaml"
ADVERSARIAL_CASES_PATH = ROOT / "fixtures" / "datasets" / "adversarial_conversations.yaml"
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


def build_categories_payload(results: list[EvaluationResult]) -> dict[str, object]:
    categories_data: dict[str, object] = {}
    categories = collect_categories(results)
    for category in categories:
        categories_data[category] = count_category_stats(results, category)
    return categories_data


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
        "suites": {
            "baseline": count_suite_stats(results, "baseline"),
            "adversarial": count_suite_stats(results, "adversarial"),
        },
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
    baseline_stats = count_suite_stats(results, "baseline")
    adversarial_stats = count_suite_stats(results, "adversarial")
    overall_passed = count_passed(results)
    overall_failed = len(results) - overall_passed

    sections: list[str] = [
        "# RAG Quality Evaluation Report",
        "",
        "## Summary",
        "",
        "| Suite | Total | Passed | Failed |",
        "| --- | ---: | ---: | ---: |",
        (
            f"| Baseline | {baseline_stats['total']} | "
            f"{baseline_stats['passed']} | {baseline_stats['failed']} |"
        ),
        (
            f"| Adversarial | {adversarial_stats['total']} | "
            f"{adversarial_stats['passed']} | {adversarial_stats['failed']} |"
        ),
        f"| **Overall** | **{len(results)}** | **{overall_passed}** | **{overall_failed}** |",
        "",
        "## Category Breakdown",
        "",
        "| Category | Total | Passed | Failed |",
        "| --- | ---: | ---: | ---: |",
    ]

    categories = collect_categories(results)
    for category in categories:
        category_stats = count_category_stats(results, category)
        sections.append(
            f"| {category} | {category_stats['total']} | "
            f"{category_stats['passed']} | {category_stats['failed']} |"
        )

    sections.append("")
    sections.append("## Baseline Results")
    sections.append("")

    baseline_results = filter_results_by_suite(results, "baseline")
    for result in baseline_results:
        append_result_section(sections, result)

    sections.append("## Adversarial Results")
    sections.append("")

    adversarial_results = filter_results_by_suite(results, "adversarial")
    for result in adversarial_results:
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


def main() -> int:
    client = AnythingLLMClient()
    workspace_slug: str | None = None
    exit_code = 1

    try:
        print("Preparing temporary workspace...")
        workspace_slug = prepare_workspace(client, STANDARD_DOCUMENTS)
        print(f"Workspace ready: {workspace_slug}")
        print()

        results: list[EvaluationResult] = []

        print("Baseline evaluations")
        print("--------------------")
        baseline_cases = load_rag_cases(BASELINE_CASES_PATH)
        for baseline_case in baseline_cases:
            print(f"Evaluating {baseline_case.id}...")
            result = evaluate_rag_case(
                client=client,
                workspace_slug=workspace_slug,
                case=baseline_case,
            )
            print_status(result)
            results.append(result)

        print()
        print("Adversarial evaluations")
        print("-----------------------")
        adversarial_cases = load_adversarial_cases(ADVERSARIAL_CASES_PATH)
        for adversarial_case in adversarial_cases:
            print(f"Evaluating {adversarial_case.id}...")
            result = evaluate_adversarial_case(
                client=client,
                workspace_slug=workspace_slug,
                case=adversarial_case,
            )
            print_status(result)
            results.append(result)

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        write_json_report(JSON_REPORT_PATH, results)
        write_markdown_report(MARKDOWN_REPORT_PATH, results)

        baseline_stats = count_suite_stats(results, "baseline")
        adversarial_stats = count_suite_stats(results, "adversarial")
        passed_count = count_passed(results)
        failed_count = len(results) - passed_count

        print()
        print(f"baseline:    {baseline_stats['passed']} passed / {baseline_stats['failed']} failed")
        print(
            f"adversarial: {adversarial_stats['passed']} passed / "
            f"{adversarial_stats['failed']} failed"
        )
        print(f"overall:     {passed_count} passed / {failed_count} failed")
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
        if workspace_slug is not None:
            print()
            print(f"Cleaning up workspace: {workspace_slug}")
            cleanup_workspace(client, workspace_slug)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
