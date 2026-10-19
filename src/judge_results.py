from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast


@dataclass(frozen=True)
class JudgeScore:
    groundedness: int
    conflict_handling: int
    unsupported_precedence: int
    answer_correctness: int
    rationale: str


JUDGE_PASS_THRESHOLD = 7

_SCORE_FIELDS = (
    "groundedness",
    "conflict_handling",
    "unsupported_precedence",
    "answer_correctness",
)


def total_score(score: JudgeScore) -> int:
    return (
        score.groundedness
        + score.conflict_handling
        + score.unsupported_precedence
        + score.answer_correctness
    )


def judge_passed(score: JudgeScore, threshold: int = JUDGE_PASS_THRESHOLD) -> bool:
    return total_score(score) >= threshold


def _require_score_int(payload: dict[str, object], key: str) -> int:
    value = payload.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"expected int field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    if value < 0 or value > 2:
        msg = f"score field {key!r} must be in 0..2, got {value}"
        raise ValueError(msg)
    return value


def _require_non_empty_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    if value.strip() == "":
        msg = f"field {key!r} must be a non-empty string"
        raise ValueError(msg)
    return value


def parse_judge_score(text: str) -> JudgeScore:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"judge output is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"expected JSON object, got {type(raw).__name__}"
        raise TypeError(msg)

    payload: dict[str, object] = {}
    for key, item in cast(list[tuple[object, object]], list(raw.items())):
        if not isinstance(key, str):
            msg = f"expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        payload[key] = item

    for field in _SCORE_FIELDS:
        if field not in payload:
            msg = f"missing required field {field!r}"
            raise ValueError(msg)
    if "rationale" not in payload:
        msg = "missing required field 'rationale'"
        raise ValueError(msg)

    return JudgeScore(
        groundedness=_require_score_int(payload, "groundedness"),
        conflict_handling=_require_score_int(payload, "conflict_handling"),
        unsupported_precedence=_require_score_int(payload, "unsupported_precedence"),
        answer_correctness=_require_score_int(payload, "answer_correctness"),
        rationale=_require_non_empty_string(payload, "rationale"),
    )
