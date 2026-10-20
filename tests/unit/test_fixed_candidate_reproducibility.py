"""Static checks: long-form runners must not require ignored reports/ inputs."""

from __future__ import annotations

from pathlib import Path

from claim_level_config import FIXED_CANDIDATE_PATH, FIXED_CANDIDATE_RELATIVE_PATH
from fixed_candidate_loader import load_fixed_candidate
from workspace_setup import PROJECT_ROOT

STALE_TRACE_MARKER = "20261019T235938"
PREVIOUS_TRACE_MARKER = "PREVIOUS_TRACE_PATH"

RUNNER_MODULES = (
    "run_split_claim_pipeline.py",
    "run_long_form_judge_diagnostic.py",
    "run_long_form_judge_qwen_16k.py",
    "run_long_form_judge_structured.py",
)


def test_fixed_candidate_path_is_under_fixtures_not_reports() -> None:
    assert FIXED_CANDIDATE_RELATIVE_PATH.startswith("fixtures/candidates/")
    assert "reports/" not in FIXED_CANDIDATE_RELATIVE_PATH
    assert FIXED_CANDIDATE_PATH.is_relative_to(PROJECT_ROOT / "fixtures")
    assert not FIXED_CANDIDATE_PATH.is_relative_to(PROJECT_ROOT / "reports")


def test_runners_have_no_hardcoded_ignored_trace_input() -> None:
    src_root = PROJECT_ROOT / "src"
    for name in RUNNER_MODULES:
        source = (src_root / name).read_text(encoding="utf-8")
        assert STALE_TRACE_MARKER not in source, f"{name} still references {STALE_TRACE_MARKER}"
        assert PREVIOUS_TRACE_MARKER not in source, f"{name} still defines {PREVIOUS_TRACE_MARKER}"
        assert "load_fixed_candidate" in source, f"{name} must load the tracked fixture"
        assert "load_identical_candidate_answer" not in source, (
            f"{name} must not parse historical traces for input"
        )


def test_src_has_no_stale_trace_timestamp_dependencies() -> None:
    src_root = PROJECT_ROOT / "src"
    offenders: list[str] = []
    for path in sorted(src_root.rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        if STALE_TRACE_MARKER in text or PREVIOUS_TRACE_MARKER in text:
            offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert offenders == []


def test_load_fixed_candidate_independent_of_reports_directory(tmp_path: Path) -> None:
    """Core clean-clone regression: loading never reads or requires reports/."""
    reports = tmp_path / "reports"
    assert not reports.exists()
    candidate = load_fixed_candidate()
    assert len(candidate) > 0
    assert not reports.exists()
    # Canonical path must remain the tracked fixture, not a reports artifact.
    assert FIXED_CANDIDATE_PATH.resolve().is_relative_to((PROJECT_ROOT / "fixtures").resolve())
