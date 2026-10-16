#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from adversarial_cases import load_adversarial_cases  # noqa: E402
from clients.anythingllm_client import AnythingLLMClient  # noqa: E402
from evaluation_results import EvaluationResult  # noqa: E402
from evaluators.adversarial_evaluator import evaluate_adversarial_case  # noqa: E402

CASES_PATH = ROOT / "fixtures" / "datasets" / "adversarial_conversations.yaml"
REPORTS_DIR = ROOT / "reports"
JSON_REPORT_PATH = REPORTS_DIR / "adversarial_report.json"
MARKDOWN_REPORT_PATH = REPORTS_DIR / "adversarial_report.md"


def count_passed(results: list[EvaluationResult]) -> int:
    passed_count = 0
    for result in results:
        if result.passed:
            passed_count += 1
    return passed_count


def result_to_json_object(result: EvaluationResult) -> dict[str, object]:
    return {
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


def build_markdown_report(results: list[EvaluationResult]) -> str:
    passed_count = count_passed(results)
    failed_count = len(results) - passed_count

    sections: list[str] = [
        "# Adversarial RAG Evaluation Report",
        "",
        "## Summary",
        "",
        "| Total | Passed | Failed |",
        "| ---: | ---: | ---: |",
        f"| {len(results)} | {passed_count} | {failed_count} |",
        "",
        "## Results",
        "",
    ]

    for result in results:
        status = "PASS" if result.passed else "FAIL"
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

    return "\n".join(sections)


def write_json_report(path: Path, results: list[EvaluationResult]) -> None:
    payload = build_json_payload(results)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
        file.write("\n")


def write_markdown_report(path: Path, results: list[EvaluationResult]) -> None:
    content = build_markdown_report(results)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    cases = load_adversarial_cases(CASES_PATH)
    client = AnythingLLMClient()
    workspace_slug = os.environ["ANYTHINGLLM_WORKSPACE"]

    results: list[EvaluationResult] = []
    for case in cases:
        print(f"Evaluating {case.id}...")
        result = evaluate_adversarial_case(
            client=client,
            workspace_slug=workspace_slug,
            case=case,
        )
        status = "PASS" if result.passed else "FAIL"
        print(f"  {status}")
        results.append(result)

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    write_json_report(JSON_REPORT_PATH, results)
    write_markdown_report(MARKDOWN_REPORT_PATH, results)

    passed_count = count_passed(results)
    failed_count = len(results) - passed_count
    print()
    print(f"Summary: {passed_count} passed / {failed_count} failed / {len(results)} total")
    print(f"Wrote {JSON_REPORT_PATH}")
    print(f"Wrote {MARKDOWN_REPORT_PATH}")

    if failed_count > 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
