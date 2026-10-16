import os
import uuid
from pathlib import Path

import pytest

from adversarial_cases import AdversarialCase, load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient

ADVERSARIAL_CASES = load_adversarial_cases(Path("fixtures/datasets/adversarial_conversations.yaml"))

ADVERSARIAL_CASE_IDS: list[str] = []
for case in ADVERSARIAL_CASES:
    ADVERSARIAL_CASE_IDS.append(case.id)


def normalize_answer_text(text: str) -> str:
    return text.replace("**", "")


def build_failure_context(case: AdversarialCase, final_answer: str) -> str:
    return (
        f"case: {case.id}\n"
        f"category: {case.category}\n"
        f"expected behavior: {case.expected_behavior}\n"
        f"failure: {case.failure_description}\n"
        f"final answer:\n{final_answer}"
    )


def contains_any_expected(text: str, expected_values: list[str]) -> bool:
    for expected in expected_values:
        if expected in text:
            return True
    return False


@pytest.mark.parametrize(
    "case",
    ADVERSARIAL_CASES,
    ids=ADVERSARIAL_CASE_IDS,
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
    failure_context = build_failure_context(case, normalized_answer)

    for expected in case.expected_final_contains_all:
        assert expected in normalized_answer, (
            f"missing expected string {expected!r}\n{failure_context}"
        )

    assert contains_any_expected(
        normalized_answer,
        case.expected_final_contains_any,
    ), f"final answer missing any of {case.expected_final_contains_any!r}\n{failure_context}"

    for banned in case.expected_final_excludes:
        assert banned not in normalized_answer, (
            f"final answer must not contain {banned!r}\n{failure_context}"
        )
