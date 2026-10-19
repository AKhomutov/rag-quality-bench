from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

ALLOWED_FACT_ASSESSMENTS = frozenset(
    {
        "supported",
        "contradicted",
        "not_addressed",
        "misattributed",
    }
)

_SCORE_FIELDS = (
    "groundedness",
    "source_authority",
    "unsupported_reasoning",
    "answer_correctness",
)


@dataclass(frozen=True)
class JudgeDimension:
    score: int
    evidence: str


@dataclass(frozen=True)
class JudgeFactCheck:
    fact: str
    candidate_position: str
    assessment: str


@dataclass(frozen=True)
class JudgeScore:
    groundedness: JudgeDimension
    source_authority: JudgeDimension
    unsupported_reasoning: JudgeDimension
    answer_correctness: JudgeDimension
    fact_checks: list[JudgeFactCheck]
    rationale: str


JUDGE_PASS_THRESHOLD = 7


def total_score(score: JudgeScore) -> int:
    return (
        score.groundedness.score
        + score.source_authority.score
        + score.unsupported_reasoning.score
        + score.answer_correctness.score
    )


def judge_passed(score: JudgeScore, threshold: int = JUDGE_PASS_THRESHOLD) -> bool:
    return total_score(score) >= threshold


def _require_object(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        msg = f"expected object field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for nested_key, item in cast(list[tuple[object, object]], list(value.items())):
        if not isinstance(nested_key, str):
            msg = f"expected string keys in {key!r}, got {type(nested_key).__name__}"
            raise TypeError(msg)
        result[nested_key] = item
    return result


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


def _parse_dimension(payload: dict[str, object], key: str) -> JudgeDimension:
    dimension = _require_object(payload, key)
    return JudgeDimension(
        score=_require_score_int(dimension, "score"),
        evidence=_require_non_empty_string(dimension, "evidence"),
    )


def _parse_fact_check(item: object, index: int) -> JudgeFactCheck:
    if not isinstance(item, dict):
        msg = f"fact_checks[{index}] must be an object, got {type(item).__name__}"
        raise TypeError(msg)
    payload: dict[str, object] = {}
    for key, value in cast(list[tuple[object, object]], list(item.items())):
        if not isinstance(key, str):
            msg = f"fact_checks[{index}] expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        payload[key] = value

    assessment = _require_non_empty_string(payload, "assessment")
    if assessment not in ALLOWED_FACT_ASSESSMENTS:
        msg = (
            f"fact_checks[{index}].assessment must be one of "
            f"{sorted(ALLOWED_FACT_ASSESSMENTS)}, got {assessment!r}"
        )
        raise ValueError(msg)

    return JudgeFactCheck(
        fact=_require_non_empty_string(payload, "fact"),
        candidate_position=_require_non_empty_string(payload, "candidate_position"),
        assessment=assessment,
    )


def _parse_fact_checks(payload: dict[str, object]) -> list[JudgeFactCheck]:
    value = payload.get("fact_checks")
    if not isinstance(value, list):
        msg = f"expected list field 'fact_checks', got {type(value).__name__}"
        raise TypeError(msg)
    if not value:
        msg = "fact_checks must be a non-empty list"
        raise ValueError(msg)

    checks: list[JudgeFactCheck] = []
    for index, item in enumerate(cast(list[object], value)):
        checks.append(_parse_fact_check(item, index))
    return checks


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
    if "fact_checks" not in payload:
        msg = "missing required field 'fact_checks'"
        raise ValueError(msg)
    if "rationale" not in payload:
        msg = "missing required field 'rationale'"
        raise ValueError(msg)

    return JudgeScore(
        groundedness=_parse_dimension(payload, "groundedness"),
        source_authority=_parse_dimension(payload, "source_authority"),
        unsupported_reasoning=_parse_dimension(payload, "unsupported_reasoning"),
        answer_correctness=_parse_dimension(payload, "answer_correctness"),
        fact_checks=_parse_fact_checks(payload),
        rationale=_require_non_empty_string(payload, "rationale"),
    )
