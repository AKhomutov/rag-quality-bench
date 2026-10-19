from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import cast


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_identical_candidate_answer(trace_path: Path) -> str:
    """Load candidate_answer from a claim-level JSONL trace.

    All lines must contain the exact same candidate_answer; otherwise fail loudly.
    The returned string is not normalized.
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
