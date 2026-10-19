from __future__ import annotations

import os
from typing import cast

import requests
from dotenv import load_dotenv

load_dotenv()

DEFAULT_OLLAMA_HOST = "http://localhost:11434"
JUDGE_MODEL = "qwen3:8b"


class OllamaClient:
    def __init__(self, base_url: str | None = None) -> None:
        host = base_url if base_url is not None else os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)
        self.base_url = host.rstrip("/")

    def generate(self, model: str, prompt: str) -> str:
        options: dict[str, int] = {"temperature": 0}
        body: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": options,
        }
        response = requests.post(
            f"{self.base_url}/api/generate",
            json=body,
            timeout=180,
        )
        response.raise_for_status()
        payload = cast(object, response.json())
        if not isinstance(payload, dict):
            msg = f"expected JSON object from Ollama, got {type(payload).__name__}"
            raise TypeError(msg)

        raw_items = cast(list[tuple[object, object]], list(payload.items()))
        result: dict[str, object] = {}
        for key, item in raw_items:
            if not isinstance(key, str):
                msg = f"expected string keys, got {type(key).__name__}"
                raise TypeError(msg)
            result[key] = item

        text = result.get("response")
        if not isinstance(text, str):
            msg = f"expected string field 'response', got {type(text).__name__}"
            raise TypeError(msg)
        return text
