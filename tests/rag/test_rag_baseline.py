import os
from pathlib import Path

import pytest

from clients.anythingllm_client import AnythingLLMClient, Source
from rag_cases import RagCase, load_rag_cases

RAG_CASES = load_rag_cases(Path("fixtures/datasets/rag_cases.yaml"))


def contains_source_text(sources: list[Source], expected_text: str) -> bool:
    for source in sources:
        if expected_text in source.text:
            return True

    return False


def normalize_answer_text(text: str) -> str:
    return text.replace("**", "")


@pytest.mark.parametrize(
    "case",
    RAG_CASES,
    ids=[case.id for case in RAG_CASES],
)
def test_rag_case(case: RagCase) -> None:
    client = AnythingLLMClient()

    response = client.chat(
        workspace_slug=os.environ["ANYTHINGLLM_WORKSPACE"],
        message=case.question,
    )

    normalized_answer = normalize_answer_text(response.final_answer)
    assert any(expected in normalized_answer for expected in case.expected_answer_contains_any)
    if case.expected_source_contains is not None:
        assert contains_source_text(
            response.sources,
            case.expected_source_contains,
        )
    if case.expected_answer_excludes is not None:
        assert case.expected_answer_excludes not in normalized_answer
