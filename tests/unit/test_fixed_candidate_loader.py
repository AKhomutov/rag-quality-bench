from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from claim_level_config import (
    EXPECTED_FIXED_CANDIDATE_LENGTH,
    EXPECTED_FIXED_CANDIDATE_SHA256,
    FIXED_CANDIDATE_PATH,
    FIXED_CANDIDATE_RELATIVE_PATH,
)
from fixed_candidate_loader import (
    assert_fixed_candidate_identity,
    load_fixed_candidate,
    load_identical_candidate_answer,
    sha256_text,
)
from workspace_setup import PROJECT_ROOT


def test_canonical_fixed_candidate_fixture_exists() -> None:
    assert FIXED_CANDIDATE_PATH.is_file()
    assert FIXED_CANDIDATE_PATH == PROJECT_ROOT / FIXED_CANDIDATE_RELATIVE_PATH


def test_load_fixed_candidate_matches_pinned_identity() -> None:
    candidate = load_fixed_candidate()
    assert len(candidate) == EXPECTED_FIXED_CANDIDATE_LENGTH
    assert sha256_text(candidate) == EXPECTED_FIXED_CANDIDATE_SHA256
    assert_fixed_candidate_identity(candidate)


def test_load_fixed_candidate_rejects_sha_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "candidate.txt"
    path.write_text("corrupted candidate content", encoding="utf-8")
    with pytest.raises(ValueError, match="candidate SHA-256 mismatch"):
        load_fixed_candidate(path)


def test_load_fixed_candidate_rejects_length_mismatch(tmp_path: Path) -> None:
    path = tmp_path / "short.txt"
    path.write_text("short", encoding="utf-8")

    def _matching_sha(_text: str) -> str:
        return EXPECTED_FIXED_CANDIDATE_SHA256

    with (
        patch("fixed_candidate_loader.sha256_text", side_effect=_matching_sha),
        pytest.raises(ValueError, match="candidate length mismatch"),
    ):
        load_fixed_candidate(path)


def test_load_fixed_candidate_missing_file(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist.txt"
    with pytest.raises(FileNotFoundError, match="fixed candidate fixture not found"):
        load_fixed_candidate(missing)


def test_load_identical_candidate_answer(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    answer = "fixed answer\nwith newline"
    path.write_text(
        '{"candidate_answer":"fixed answer\\nwith newline","trial_number":1}\n'
        '{"candidate_answer":"fixed answer\\nwith newline","trial_number":2}\n',
        encoding="utf-8",
    )
    loaded = load_identical_candidate_answer(path)
    assert loaded == answer
    assert sha256_text(loaded) == sha256_text(answer)


def test_load_rejects_mismatched_candidates(tmp_path: Path) -> None:
    path = tmp_path / "trace.jsonl"
    path.write_text(
        '{"candidate_answer":"a"}\n{"candidate_answer":"b"}\n',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="candidate_answer mismatch"):
        load_identical_candidate_answer(path)
