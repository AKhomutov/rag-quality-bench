from __future__ import annotations

from pathlib import Path

import pytest

from fixed_candidate_loader import load_identical_candidate_answer, sha256_text


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
