import os
import uuid
from pathlib import Path

import pytest

from adversarial_cases import AdversarialCase, load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient

ADVERSARIAL_CASES = load_adversarial_cases(Path("fixtures/datasets/adversarial_conversations.yaml"))


def normalize_answer_text(text: str) -> str:
    return text.replace("**", "")


@pytest.mark.parametrize(
    "case",
    ADVERSARIAL_CASES,
    ids=[case.id for case in ADVERSARIAL_CASES],
)
def test_adversarial_conversation(case: AdversarialCase) -> None:
    client = AnythingLLMClient()
    workspace_slug = os.environ["ANYTHINGLLM_WORKSPACE"]
    thread_slug = f"{case.id}-{uuid.uuid4().hex[:8]}"

    created_slug = client.create_thread(workspace_slug, thread_slug)

    final_response = None
    for step in case.steps:
        final_response = client.thread_chat(
            workspace_slug=workspace_slug,
            thread_slug=created_slug,
            message=step.message,
        )

    assert final_response is not None
    normalized_answer = normalize_answer_text(final_response.final_answer)

    for expected in case.expected_final_contains_all:
        assert expected in normalized_answer, (
            f"case={case.id}: missing expected string {expected!r}:\n{normalized_answer}"
        )

    assert any(expected in normalized_answer for expected in case.expected_final_contains_any), (
        f"case={case.id}: final answer missing expected content:\n{normalized_answer}"
    )

    for banned in case.expected_final_excludes:
        assert banned not in normalized_answer, (
            f"case={case.id}: final answer must not contain {banned!r}:\n{normalized_answer}"
        )
