"""Unit tests for opt-in long-form claim audit integration."""

from __future__ import annotations

from pathlib import Path
from typing import cast
from unittest.mock import Mock

import pytest

from evaluation_results import EvaluationResult
from long_form_claim_audit_results import (
    LONG_FORM_SUITE,
    LongFormClaimAuditResult,
    evaluation_status,
    long_form_result_to_evaluation_result,
)
from run_evaluations import (
    build_json_payload,
    build_markdown_report,
    count_failed,
    maybe_run_long_form_suite,
    parse_evaluation_args,
    result_to_json_object,
)


def _sample_audit(*, status: str, execution_error: str | None = None) -> LongFormClaimAuditResult:
    return LongFormClaimAuditResult(
        case_id="mixed_truth_severity_policy_poisoning",
        status=status,
        total_claims=54,
        supported_count=28,
        contradicted_count=22,
        not_in_evidence_count=2,
        derived_count=2,
        provenance_issue_count=20,
        grounded_claim_rate=0.519,
        raw_dependency_edge_count=17,
        normalized_dependency_edge_count=15,
        dependency_cycle_count=1,
        execution_error=execution_error,
        duration_seconds=12.5,
        debug_path="reports/judge_debug/example",
        judge_model="qwen3:8b",
        judge_model_digest="sha256:deadbeef",
        num_ctx=16384,
        candidate_sha256="abc123",
        prompt_sha256="def456",
        failure_summary=(
            []
            if status == "PASS"
            else (
                [execution_error]
                if execution_error is not None
                else ["claim C01 status='contradicted': example"]
            )
        ),
    )


def _baseline_result(*, passed: bool = True) -> EvaluationResult:
    return EvaluationResult(
        suite="baseline",
        case_id="simple_lookup",
        category="factual",
        passed=passed,
        expected_behavior="answer from docs",
        failure_description="wrong answer",
        final_answer="ok",
        failures=[] if passed else ["mismatch"],
        tools_used=[],
        trial_summary=None,
    )


def test_parse_args_default_excludes_long_form() -> None:
    options = parse_evaluation_args([])
    assert options.include_long_form is False


def test_parse_args_include_long_form() -> None:
    options = parse_evaluation_args(["--include-long-form"])
    assert options.include_long_form is True


def test_parse_args_rejects_unknown_flag() -> None:
    with pytest.raises(SystemExit):
        parse_evaluation_args(["--unknown-flag"])


def test_default_does_not_call_split_runner() -> None:
    results: list[EvaluationResult] = [_baseline_result()]
    runner = Mock()
    maybe_run_long_form_suite(results, include_long_form=False, runner=runner)
    runner.assert_not_called()
    assert len(results) == 1
    assert all(result.suite != LONG_FORM_SUITE for result in results)


def test_opt_in_calls_split_runner_once_and_maps_dto() -> None:
    results: list[EvaluationResult] = [_baseline_result()]
    audit = _sample_audit(status="FAIL")
    runner = Mock(return_value=audit)

    maybe_run_long_form_suite(results, include_long_form=True, runner=runner)

    runner.assert_called_once_with()
    assert len(results) == 2
    long_form = results[-1]
    assert long_form.suite == LONG_FORM_SUITE
    assert long_form.case_id == "mixed_truth_severity_policy_poisoning"
    assert long_form.passed is False
    assert long_form.execution_error is None
    assert long_form.long_form_metrics is not None
    assert long_form.long_form_metrics["total_claims"] == 54
    assert long_form.debug_path == "reports/judge_debug/example"


def test_status_pass_mapping() -> None:
    result = long_form_result_to_evaluation_result(_sample_audit(status="PASS"))
    assert result.passed is True
    assert result.execution_error is None
    assert evaluation_status(result) == "PASS"


def test_status_fail_mapping() -> None:
    result = long_form_result_to_evaluation_result(_sample_audit(status="FAIL"))
    assert result.passed is False
    assert result.execution_error is None
    assert evaluation_status(result) == "FAIL"


def test_status_error_mapping() -> None:
    result = long_form_result_to_evaluation_result(
        _sample_audit(status="ERROR", execution_error="candidate SHA mismatch")
    )
    assert result.passed is False
    assert result.execution_error == "candidate SHA mismatch"
    assert evaluation_status(result) == "ERROR"


def test_json_report_includes_long_form_suite() -> None:
    results = [
        _baseline_result(),
        long_form_result_to_evaluation_result(_sample_audit(status="FAIL")),
    ]
    payload = build_json_payload(results)
    suites = cast(dict[str, object], payload["suites"])
    assert LONG_FORM_SUITE in suites
    results_raw = cast(list[object], payload["results"])
    long_form_entries: list[dict[str, object]] = []
    for item in results_raw:
        entry = cast(dict[str, object], item)
        if entry["suite"] == LONG_FORM_SUITE:
            long_form_entries.append(entry)
    assert len(long_form_entries) == 1
    entry = long_form_entries[0]
    assert entry["status"] == "FAIL"
    assert entry["case_id"] == "mixed_truth_severity_policy_poisoning"
    metrics = cast(dict[str, object], entry["metrics"])
    dependency_graph = cast(dict[str, object], entry["dependency_graph"])
    assert metrics["total_claims"] == 54
    assert metrics["supported"] == 28
    assert dependency_graph["raw_edges"] == 17
    assert dependency_graph["normalized_edges"] == 15
    assert dependency_graph["cycles"] == 1
    assert entry["debug_path"] == "reports/judge_debug/example"
    assert "all claims" not in str(entry).lower()


def test_markdown_report_includes_long_form_section_only_when_present() -> None:
    without = build_markdown_report([_baseline_result()])
    assert "## Long-form claim audit" not in without

    with_long = build_markdown_report(
        [
            _baseline_result(),
            long_form_result_to_evaluation_result(_sample_audit(status="FAIL")),
        ]
    )
    assert "## Long-form claim audit" in with_long
    assert "Case: mixed_truth_severity_policy_poisoning" in with_long
    assert "Status: FAIL" in with_long
    assert "Claims: 54" in with_long
    assert "Detailed trace:" in with_long
    assert "reports/judge_debug/example" in with_long
    # Concise report: no full claim dump.
    assert "claim C01 status=" in with_long  # representative failure ok
    assert with_long.count("claim C") <= 5


def test_default_json_has_no_long_form_suite() -> None:
    payload = build_json_payload([_baseline_result()])
    suites = cast(dict[str, object], payload["suites"])
    assert LONG_FORM_SUITE not in suites
    results_raw = cast(list[object], payload["results"])
    for item in results_raw:
        entry = cast(dict[str, object], item)
        assert entry["suite"] != LONG_FORM_SUITE


def test_long_form_fail_affects_overall_outcome() -> None:
    results = [
        _baseline_result(passed=True),
        long_form_result_to_evaluation_result(_sample_audit(status="FAIL")),
    ]
    assert count_failed(results) == 1
    exit_would_be = 1 if count_failed(results) > 0 else 0
    assert exit_would_be == 1


def test_long_form_error_affects_overall_outcome() -> None:
    results = [
        _baseline_result(passed=True),
        long_form_result_to_evaluation_result(
            _sample_audit(status="ERROR", execution_error="parse failure")
        ),
    ]
    assert count_failed(results) == 1
    assert evaluation_status(results[-1]) == "ERROR"
    exit_would_be = 1 if count_failed(results) > 0 else 0
    assert exit_would_be == 1


def test_result_to_json_object_error_includes_execution_error() -> None:
    result = long_form_result_to_evaluation_result(
        _sample_audit(status="ERROR", execution_error="graph validation failure")
    )
    payload = result_to_json_object(result)
    assert payload["status"] == "ERROR"
    assert payload["execution_error"] == "graph validation failure"
    assert payload["passed"] is False


def test_mapper_preserves_debug_path_as_string() -> None:
    audit = _sample_audit(status="PASS")
    result = long_form_result_to_evaluation_result(audit)
    assert result.debug_path == str(Path("reports/judge_debug/example"))
