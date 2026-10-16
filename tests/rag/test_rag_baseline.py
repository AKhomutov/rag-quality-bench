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

    assert case.expected_answer_contains in response.text_response
    assert contains_source_text(
        response.sources,
        case.expected_source_contains,
    )
