"""Load the tracked fixed candidate, or extract one from a historical JSONL trace.

Tracked fixture path and expected identity live in ``claim_level_config``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast

from claim_level_config import (
    EXPECTED_FIXED_CANDIDATE_LENGTH,
    EXPECTED_FIXED_CANDIDATE_SHA256,
    FIXED_CANDIDATE_PATH,
)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def assert_fixed_candidate_identity(candidate_answer: str) -> None:
    """Validate SHA-256 and character length against the pinned contract."""
    digest = sha256_text(candidate_answer)
    if digest != EXPECTED_FIXED_CANDIDATE_SHA256:
        msg = (
            f"candidate SHA-256 mismatch: expected {EXPECTED_FIXED_CANDIDATE_SHA256}, got {digest}"
        )
        raise ValueError(msg)
    length = len(candidate_answer)
    if length != EXPECTED_FIXED_CANDIDATE_LENGTH:
        msg = f"candidate length mismatch: expected {EXPECTED_FIXED_CANDIDATE_LENGTH}, got {length}"
        raise ValueError(msg)


def load_fixed_candidate(path: Path = FIXED_CANDIDATE_PATH) -> str:
    """Load the versioned fixed candidate fixture and validate identity.

    Reads UTF-8 with no whitespace normalization. Fails loudly if the file is
    missing or if SHA-256 / length do not match the pinned contract.
    """
    if not path.is_file():
        msg = f"fixed candidate fixture not found: {path}"
        raise FileNotFoundError(msg)
    candidate_answer = path.read_text(encoding="utf-8")
    assert_fixed_candidate_identity(candidate_answer)
    return candidate_answer


def load_identical_candidate_answer(trace_path: Path) -> str:
    """Historical helper: load candidate_answer from a claim-level JSONL trace.

    All lines must contain the exact same candidate_answer; otherwise fail loudly.
    The returned string is not normalized and is not SHA-validated — callers that
    need the pinned identity should call ``assert_fixed_candidate_identity``.

    Not used by the current split pipeline or long-form judge runners.
    """
    if not trace_path.is_file():
        msg = f"candidate source trace not found: {trace_path}"
        raise FileNotFoundError(msg)

    answers: list[str] = []
    with trace_path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if line.strip() == "":
                continue
            raw: object = json.loads(line)
            if not isinstance(raw, dict):
                msg = f"{trace_path}:{line_number}: expected JSON object, got {type(raw).__name__}"
                raise TypeError(msg)
            payload: dict[str, object] = {}
            for key, value in cast(list[tuple[object, object]], list(raw.items())):
                if not isinstance(key, str):
                    msg = f"{trace_path}:{line_number}: expected string keys"
                    raise TypeError(msg)
                payload[key] = value
            candidate = payload.get("candidate_answer")
            if not isinstance(candidate, str):
                msg = (
                    f"{trace_path}:{line_number}: candidate_answer must be a string, "
                    f"got {type(candidate).__name__}"
                )
                raise TypeError(msg)
            answers.append(candidate)

    if not answers:
        msg = f"no candidate answers found in {trace_path}"
        raise ValueError(msg)

    first = answers[0]
    for index, answer in enumerate(answers[1:], start=2):
        if answer != first:
            msg = (
                f"candidate_answer mismatch in {trace_path}: "
                f"line 1 sha={sha256_text(first)} "
                f"line {index} sha={sha256_text(answer)}"
            )
            raise ValueError(msg)

    return first
