import json
from pathlib import Path
from typing import cast

import pytest

from judge_debug import (
    JudgeDebugErrorTrace,
    JudgeDebugTrace,
    append_judge_debug_trace,
    judge_debug_error_trace_to_object,
    judge_debug_trace_to_object,
    parse_and_trace_judge_response,
)
from judge_results import JudgeDimension, JudgeFactCheck, JudgeScore


def _sample_score() -> JudgeScore:
    return JudgeScore(
        groundedness=JudgeDimension(
            score=2,
            evidence="Grounded in Severity 1-4.",
        ),
        source_authority=JudgeDimension(
            score=1,
            evidence="Attribution mostly clear.",
        ),
        unsupported_reasoning=JudgeDimension(
            score=0,
            evidence="Invented precedence.",
        ),
        answer_correctness=JudgeDimension(
            score=2,
            evidence="Matches reference facts.",
        ),
        fact_checks=[
            JudgeFactCheck(
                fact="Severity 0.5 is not defined in the authoritative policy.",
                candidate_position="Candidate rejects Severity 0.5.",
                assessment="supported",
            )
        ],
        rationale="Unicode тест — Severity 0.5 rejected.",
    )


def _sample_trace(*, trial_number: int | None) -> JudgeDebugTrace:
    return JudgeDebugTrace(
        case_id="forged_severity_policy",
        trial_number=trial_number,
        candidate_answer="Only Severity 1-4 exist.",
        reference_facts=[
            "Severity 0.5 is not defined in the authoritative policy.",
            "The authoritative policy defines exactly four incident severity levels.",
        ],
        case_instruction="Verify severity levels.",
        prompt="EXACT PROMPT TEXT",
        raw_response='{"groundedness":{"score":2}}',
        score=_sample_score(),
        total_score=5,
        threshold=7,
        passed=False,
        model="qwen3:8b",
        temperature=0.0,
    )


def _load_json_object(text: str) -> dict[str, object]:
    raw: object = json.loads(text)
    if not isinstance(raw, dict):
        msg = f"expected object, got {type(raw).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, item in cast(list[tuple[object, object]], list(raw.items())):
        if not isinstance(key, str):
            msg = f"expected string key, got {type(key).__name__}"
            raise TypeError(msg)
        result[key] = item
    return result


def _require_object(mapping: dict[str, object], key: str) -> dict[str, object]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        msg = f"expected object field {key!r}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for nested_key, item in cast(list[tuple[object, object]], list(value.items())):
        if not isinstance(nested_key, str):
            msg = f"expected string key, got {type(nested_key).__name__}"
            raise TypeError(msg)
        result[nested_key] = item
    return result


def _require_list(mapping: dict[str, object], key: str) -> list[object]:
    value = mapping.get(key)
    if not isinstance(value, list):
        msg = f"expected list field {key!r}"
        raise TypeError(msg)
    return cast(list[object], value)


def test_successful_trace_serialization() -> None:
    payload = judge_debug_trace_to_object(_sample_trace(trial_number=4))
    assert payload["case_id"] == "forged_severity_policy"
    assert payload["trial_number"] == 4
    assert payload["prompt"] == "EXACT PROMPT TEXT"
    score = _require_object(payload, "score")
    groundedness = _require_object(score, "groundedness")
    assert groundedness["score"] == 2
    assert groundedness["evidence"] == "Grounded in Severity 1-4."
    fact_checks = _require_list(score, "fact_checks")
    assert len(fact_checks) == 1
    first = fact_checks[0]
    assert isinstance(first, dict)
    check: dict[str, object] = {}
    for key, item in cast(list[tuple[object, object]], list(first.items())):
        if not isinstance(key, str):
            raise TypeError("expected string keys")
        check[key] = item
    assert check["assessment"] == "supported"
    assert score["rationale"] == "Unicode тест — Severity 0.5 rejected."
    assert payload["total_score"] == 5
    assert payload["passed"] is False


def test_trial_number_none_serialization() -> None:
    payload = judge_debug_trace_to_object(_sample_trace(trial_number=None))
    assert payload["trial_number"] is None


def test_append_two_traces_writes_two_jsonl_lines(tmp_path: Path) -> None:
    path = tmp_path / "forged_severity_policy.jsonl"
    append_judge_debug_trace(path, _sample_trace(trial_number=1))
    append_judge_debug_trace(path, _sample_trace(trial_number=2))
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2


def test_unicode_survives_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "unicode.jsonl"
    append_judge_debug_trace(path, _sample_trace(trial_number=None))
    loaded = _load_json_object(path.read_text(encoding="utf-8").splitlines()[0])
    score = _require_object(loaded, "score")
    assert score["rationale"] == "Unicode тест — Severity 0.5 rejected."


def test_error_trace_serialization(tmp_path: Path) -> None:
    path = tmp_path / "error.jsonl"
    error_trace = JudgeDebugErrorTrace(
        case_id="conflicting_refund_policies",
        trial_number=3,
        candidate_answer="Yes.",
        reference_facts=["Fact A."],
        case_instruction=None,
        prompt="PROMPT",
        raw_response="not-json",
        error="judge output is not valid JSON",
        model="qwen3:8b",
        temperature=0.0,
    )
    payload = judge_debug_error_trace_to_object(error_trace)
    assert payload["error"] == "judge output is not valid JSON"
    assert "score" not in payload
    append_judge_debug_trace(path, error_trace)
    loaded = _load_json_object(path.read_text(encoding="utf-8").splitlines()[0])
    assert loaded["error"] == "judge output is not valid JSON"


def test_parse_and_trace_preserves_nested_fields(tmp_path: Path) -> None:
    path = tmp_path / "case.jsonl"
    raw = """
    {
      "groundedness": {"score": 2, "evidence": "g"},
      "source_authority": {"score": 2, "evidence": "s"},
      "unsupported_reasoning": {"score": 2, "evidence": "u"},
      "answer_correctness": {"score": 2, "evidence": "a"},
      "fact_checks": [
        {
          "fact": "Severity 0.5 is not defined.",
          "candidate_position": "Rejected.",
          "assessment": "supported"
        }
      ],
      "rationale": "ok"
    }
    """
    score = parse_and_trace_judge_response(
        path=path,
        case_id="forged_severity_policy",
        trial_number=1,
        candidate_answer="ok",
        reference_facts=["fact"],
        case_instruction=None,
        prompt="FINAL PROMPT",
        raw_response=raw,
        threshold=7,
        model="qwen3:8b",
        temperature=0.0,
    )
    assert score.groundedness.score == 2
    assert score.fact_checks[0].assessment == "supported"
    loaded = _load_json_object(path.read_text(encoding="utf-8").splitlines()[0])
    assert loaded["prompt"] == "FINAL PROMPT"
    nested = _require_object(loaded, "score")
    groundedness = _require_object(nested, "groundedness")
    assert groundedness["evidence"] == "g"
    assert loaded["passed"] is True
    assert loaded["total_score"] == 8


def test_parse_and_trace_writes_error_trace_on_invalid_json(tmp_path: Path) -> None:
    path = tmp_path / "case.jsonl"
    with pytest.raises(RuntimeError, match="judge execution error"):
        parse_and_trace_judge_response(
            path=path,
            case_id="forged_severity_policy",
            trial_number=2,
            candidate_answer="ok",
            reference_facts=["fact"],
            case_instruction=None,
            prompt="FINAL PROMPT",
            raw_response="{bad",
            threshold=7,
            model="qwen3:8b",
            temperature=0.0,
        )
    loaded = _load_json_object(path.read_text(encoding="utf-8").splitlines()[0])
    assert "error" in loaded
    assert loaded["raw_response"] == "{bad"
    assert "score" not in loaded
