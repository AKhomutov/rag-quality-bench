import os
from dataclasses import dataclass
from pathlib import Path
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


def _parse_chat_response(raw: object) -> ChatResponse:
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

    def create_workspace(self, name: str, slug: str) -> str:
        body: dict[str, str] = {
            "name": name,
            "slug": slug,
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/workspace/new",
            json=body,
            timeout=60,
        )
        response.raise_for_status()

        raw: object = response.json()
        payload = _validate_json_object(raw)
        workspace_raw = payload.get("workspace")
        workspace = _validate_json_object(workspace_raw)
        return _get_required_string(workspace, "slug")

    def delete_workspace(self, workspace_slug: str) -> None:
        response = self.session.delete(
            f"{self.base_url}/api/v1/workspace/{workspace_slug}",
            timeout=60,
        )
        response.raise_for_status()

    def update_workspace(
        self,
        workspace_slug: str,
        chat_provider: str,
        chat_model: str,
    ) -> None:
        body: dict[str, str] = {
            "chatProvider": chat_provider,
            "chatModel": chat_model,
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/workspace/{workspace_slug}/update",
            json=body,
            timeout=60,
        )
        response.raise_for_status()

    def upload_document(self, path: Path) -> str:
        with path.open("rb") as file_handle:
            response = self.session.post(
                f"{self.base_url}/api/v1/document/upload",
                files={"file": (path.name, file_handle, "text/markdown")},
                headers={"Content-Type": None},
                timeout=120,
            )
        response.raise_for_status()

        raw: object = response.json()
        payload = _validate_json_object(raw)

        success = payload.get("success")
        if success is not True:
            error = payload.get("error")
            msg = f"document upload failed for {path.name!r}: {error!r}"
            raise RuntimeError(msg)

        documents_raw = payload.get("documents")
        if not isinstance(documents_raw, list) or len(documents_raw) == 0:
            msg = f"document upload returned no documents for {path.name!r}: {payload!r}"
            raise RuntimeError(msg)

        first_document = _validate_json_object(cast(list[object], documents_raw)[0])
        return _get_required_string(first_document, "location")

    def add_document_to_workspace(
        self,
        workspace_slug: str,
        document_location: str,
    ) -> None:
        body: dict[str, list[str]] = {
            "adds": [document_location],
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/workspace/{workspace_slug}/update-embeddings",
            json=body,
            timeout=180,
        )
        response.raise_for_status()

    def create_thread(self, workspace_slug: str, thread_slug: str) -> str:
        body: dict[str, str] = {
            "name": thread_slug,
            "slug": thread_slug,
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/workspace/{workspace_slug}/thread/new",
            json=body,
            timeout=60,
        )
        response.raise_for_status()

        raw: object = response.json()
        payload = _validate_json_object(raw)
        thread_raw = payload.get("thread")
        thread = _validate_json_object(thread_raw)
        return _get_required_string(thread, "slug")

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
        return _parse_chat_response(raw)

    def thread_chat(
        self,
        workspace_slug: str,
        thread_slug: str,
        message: str,
    ) -> ChatResponse:
        body: dict[str, str] = {
            "message": message,
            "mode": "query",
        }
        response = self.session.post(
            f"{self.base_url}/api/v1/workspace/{workspace_slug}/thread/{thread_slug}/chat",
            json=body,
            timeout=60,
        )
        response.raise_for_status()
        raw: object = response.json()
        return _parse_chat_response(raw)
