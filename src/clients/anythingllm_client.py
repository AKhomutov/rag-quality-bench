import os
from dataclasses import dataclass
from typing import cast

import requests
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Source:
    title: str
    text: str


@dataclass(frozen=True)
class ChatResponse:
    text_response: str
    final_answer: str
    sources: list[Source]


def _extract_final_answer(text_response: str) -> str:
    marker = "</think>"
    if marker in text_response:
        return text_response.split(marker, 1)[1].strip()
    return text_response.strip()


def _validate_json_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"expected JSON object, got {type(value).__name__}"
        raise TypeError(msg)

    result: dict[str, object] = {}
    raw_items = cast(list[tuple[object, object]], list(value.items()))
    for key, item in raw_items:
        if not isinstance(key, str):
            msg = f"expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        result[key] = item
    return result


def _get_required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)
    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


class AnythingLLMClient:
    def __init__(self) -> None:
        self.base_url = os.environ["ANYTHINGLLM_BASE_URL"].rstrip("/")
        self.api_key = os.environ["ANYTHINGLLM_API_KEY"]

        self.session = requests.Session()
        self.session.headers.update(
            {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
        )

    def chat(self, workspace_slug: str, message: str) -> ChatResponse:
        body: dict[str, str] = {
            "message": message,
            "mode": "query",
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/workspace/{workspace_slug}/chat",
            json=body,
            timeout=60,
        )
        response.raise_for_status()

        raw: object = response.json()
        payload = _validate_json_object(raw)

        sources_raw = payload.get("sources")
        if not isinstance(sources_raw, list):
            raise TypeError("expected sources list")

        sources: list[Source] = []
        for item in cast(list[object], sources_raw):
            source = _validate_json_object(item)
            sources.append(
                Source(
                    title=_get_required_string(source, "title"),
                    text=_get_required_string(source, "text"),
                )
            )

        text_response = _get_required_string(payload, "textResponse")
        return ChatResponse(
            text_response=text_response,
            final_answer=_extract_final_answer(text_response),
            sources=sources,
        )
