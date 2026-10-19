from __future__ import annotations

from typing import cast

from clients.ollama_client import build_generate_request_body
from json_types import JsonObject
from judges.claim_level_schema import build_claim_level_json_schema


def test_generate_body_without_schema_uses_json_format_string() -> None:
    body = build_generate_request_body(
        model="qwen3:8b",
        prompt="hello",
        json_schema=None,
    )
    assert body["format"] == "json"
    assert body["model"] == "qwen3:8b"
    assert body["prompt"] == "hello"
    assert body["stream"] is False
    assert "keep_alive" not in body
    options_raw = body["options"]
    assert isinstance(options_raw, dict)
    options = cast(dict[str, object], options_raw)
    temperature = options["temperature"]
    assert temperature == 0


def test_generate_body_with_schema_passes_schema_object() -> None:
    schema: JsonObject = build_claim_level_json_schema()
    body = build_generate_request_body(
        model="qwen3:8b",
        prompt="hello",
        json_schema=schema,
    )
    assert body["format"] is schema
    format_value = body["format"]
    assert isinstance(format_value, dict)
    format_object = cast(dict[str, object], format_value)
    assert format_object["type"] == "object"
    properties_raw = format_object["properties"]
    assert isinstance(properties_raw, dict)
    properties = cast(dict[str, object], properties_raw)
    assert "claims" in properties
