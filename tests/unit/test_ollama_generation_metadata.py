from __future__ import annotations

from typing import cast

import pytest

from clients.ollama_client import (
    build_generate_request_body,
    parse_ollama_generation_payload,
)


def test_parse_generation_payload_happy_path() -> None:
    result = parse_ollama_generation_payload(
        {
            "response": '{"ok": true}',
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 1200,
            "eval_count": 400,
            "total_duration": 11,
            "load_duration": 2,
            "prompt_eval_duration": 3,
            "eval_duration": 4,
        }
    )
    assert result.response == '{"ok": true}'
    assert result.done is True
    assert result.done_reason == "stop"
    assert result.prompt_eval_count == 1200
    assert result.eval_count == 400
    assert result.total_duration == 11
    assert result.load_duration == 2
    assert result.prompt_eval_duration == 3
    assert result.eval_duration == 4


def test_parse_generation_payload_optional_fields_missing() -> None:
    result = parse_ollama_generation_payload(
        {
            "response": "{}",
            "done": False,
        }
    )
    assert result.done is False
    assert result.done_reason is None
    assert result.prompt_eval_count is None
    assert result.eval_count is None
    assert result.total_duration is None
    assert result.load_duration is None
    assert result.prompt_eval_duration is None
    assert result.eval_duration is None


def test_parse_generation_requires_done() -> None:
    with pytest.raises(ValueError, match="done"):
        parse_ollama_generation_payload({"response": "{}"})


def test_num_ctx_serialized_when_supplied() -> None:
    body = build_generate_request_body(
        model="gemma3:12b",
        prompt="x",
        num_ctx=16384,
    )
    options_raw = body["options"]
    assert isinstance(options_raw, dict)
    options = cast(dict[str, object], options_raw)
    assert options["num_ctx"] == 16384
    assert options["temperature"] == 0


def test_num_ctx_omitted_when_none() -> None:
    body = build_generate_request_body(model="gemma3:12b", prompt="x", num_ctx=None)
    options_raw = body["options"]
    assert isinstance(options_raw, dict)
    options = cast(dict[str, object], options_raw)
    assert "num_ctx" not in options
