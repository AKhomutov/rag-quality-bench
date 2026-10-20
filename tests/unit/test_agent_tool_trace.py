from __future__ import annotations

from clients.anythingllm_client import _extract_tool_calls_from_thoughts
from evaluators.agent_tool_selection_evaluator import (
    any_preferred_tool_used,
    tool_was_used,
    unique_tools,
)


def test_extract_tool_calls_from_thought_lines() -> None:
    thoughts = [
        "thinking...",
        '@agent is executing `rag-memory` tool {"query":"retention"}',
        "more text",
        '@agent is executing `web-browsing` tool {"url":"https://example.com"}',
    ]
    calls = _extract_tool_calls_from_thoughts(thoughts)
    assert [call.name for call in calls] == ["rag-memory", "web-browsing"]


def test_extract_tool_calls_ignores_malformed_lines() -> None:
    thoughts = [
        "@agent is executing `` tool {}",
        "agent is executing `rag-memory` tool {}",
        "@agent is executing `rag-memory`",
    ]
    assert _extract_tool_calls_from_thoughts(thoughts) == []


def test_tool_was_used_exact_match() -> None:
    assert tool_was_used(["rag-memory", "web-browsing"], "rag-memory") is True
    assert tool_was_used(["rag-memory"], "web-browsing") is False


def test_any_preferred_tool_used() -> None:
    assert any_preferred_tool_used(["rag-memory"], ["rag-memory", "document"]) is True
    assert any_preferred_tool_used(["web-browsing"], ["rag-memory"]) is False


def test_unique_tools_preserves_order() -> None:
    assert unique_tools(["rag-memory", "web-browsing", "rag-memory"]) == [
        "rag-memory",
        "web-browsing",
    ]
