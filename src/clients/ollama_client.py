from __future__ import annotations

import os
from dataclasses import dataclass
from typing import cast

import requests
from dotenv import load_dotenv

from json_types import JsonObject

load_dotenv()

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "qwen3:8b"
JUDGE_TEMPERATURE = 0


@dataclass(frozen=True)
class OllamaGenerationResult:
    response: str
    done: bool
    done_reason: str | None
    prompt_eval_count: int | None
    eval_count: int | None
    total_duration: int | None
    load_duration: int | None
    prompt_eval_duration: int | None
    eval_duration: int | None


def build_generate_request_body(
    *,
    model: str,
    prompt: str,
    json_schema: JsonObject | None = None,
    keep_alive: int | str | None = None,
    temperature: int = JUDGE_TEMPERATURE,
    num_ctx: int | None = None,
) -> dict[str, object]:
    options: dict[str, int] = {"temperature": temperature}
    if num_ctx is not None:
        options["num_ctx"] = num_ctx
    format_value: object
    if json_schema is None:
        format_value = "json"
    else:
        format_value = json_schema
    body: dict[str, object] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": format_value,
        "options": options,
    }
    if keep_alive is not None:
        body["keep_alive"] = keep_alive
    return body


def _optional_bool(payload: dict[str, object], key: str) -> bool | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, bool):
        msg = f"expected bool field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _optional_int(payload: dict[str, object], key: str) -> int | None:
    value = payload.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"expected int field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _optional_str(payload: dict[str, object], key: str) -> str | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def parse_ollama_generation_payload(payload: dict[str, object]) -> OllamaGenerationResult:
    text = payload.get("response")
    if not isinstance(text, str):
        msg = f"expected string field 'response', got {type(text).__name__}"
        raise TypeError(msg)

    done = _optional_bool(payload, "done")
    if done is None:
        msg = "missing required bool field 'done'"
        raise ValueError(msg)

    return OllamaGenerationResult(
        response=text,
        done=done,
        done_reason=_optional_str(payload, "done_reason"),
        prompt_eval_count=_optional_int(payload, "prompt_eval_count"),
        eval_count=_optional_int(payload, "eval_count"),
        total_duration=_optional_int(payload, "total_duration"),
        load_duration=_optional_int(payload, "load_duration"),
        prompt_eval_duration=_optional_int(payload, "prompt_eval_duration"),
        eval_duration=_optional_int(payload, "eval_duration"),
    )


def payload_dict_from_response_json(raw: object) -> dict[str, object]:
    if not isinstance(raw, dict):
        msg = f"expected JSON object from Ollama, got {type(raw).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, item in cast(list[tuple[object, object]], list(raw.items())):
        if not isinstance(key, str):
            msg = f"expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        result[key] = item
    return result


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        host = base_url if base_url is not None else os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.base_url = host.rstrip("/")

    def generate_with_metadata(
        self,
        model: str,
        prompt: str,
        *,
        timeout: int = 180,
        json_schema: JsonObject | None = None,
        keep_alive: int | str | None = None,
        temperature: int = JUDGE_TEMPERATURE,
        num_ctx: int | None = None,
    ) -> OllamaGenerationResult:
        body = build_generate_request_body(
            model=model,
            prompt=prompt,
            json_schema=json_schema,
            keep_alive=keep_alive,
            temperature=temperature,
            num_ctx=num_ctx,
        )
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=body,
            timeout=timeout,
        )
        response.raise_for_status()
        payload = payload_dict_from_response_json(cast(object, response.json()))
        return parse_ollama_generation_payload(payload)

    def generate(
        self,
        model: str,
        prompt: str,
        *,
        timeout: int = 180,
        json_schema: JsonObject | None = None,
        keep_alive: int | str | None = None,
        temperature: int = JUDGE_TEMPERATURE,
        num_ctx: int | None = None,
    ) -> str:
        return self.generate_with_metadata(
            model=model,
            prompt=prompt,
            timeout=timeout,
            json_schema=json_schema,
            keep_alive=keep_alive,
            temperature=temperature,
            num_ctx=num_ctx,
        ).response

    def show_model(self, model: str) -> dict[str, object]:
        show_body: dict[str, object] = {"name": model}
        response = requests.post(
            f"{self.base_url}/api/show",
            json=show_body,
            timeout=60,
        )
        response.raise_for_status()
        raw_json: object = response.json()
        return payload_dict_from_response_json(raw_json)
