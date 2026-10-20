from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock, patch

from clients.ollama_client import OllamaClient


def test_unload_model_posts_keep_alive_zero() -> None:
    client = OllamaClient(base_url="http://localhost:11434")
    mock_response = MagicMock()
    mock_response.raise_for_status = MagicMock()

    with patch("clients.ollama_client.requests.post", return_value=mock_response) as post:
        client.unload_model("qwen3:8b")

    post.assert_called_once()
    args = cast(tuple[object, ...], post.call_args.args)
    kwargs = cast(dict[str, object], post.call_args.kwargs)
    body = cast(dict[str, object], kwargs["json"])
    assert body["model"] == "qwen3:8b"
    assert body["keep_alive"] == 0
    assert body["prompt"] == ""
    assert body["stream"] is False
    url = cast(str, args[0])
    assert url.endswith("/api/generate")
