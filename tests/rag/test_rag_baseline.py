from pathlib import Path

import pytest

from clients.anythingllm_client import AnythingLLMClient
from evaluation_assertions import format_evaluation_failure
from evaluators.baseline_evaluator import evaluate_rag_case
from rag_cases import RagCase, load_rag_cases

RAG_CASES = load_rag_cases(Path("fixtures/datasets/rag_cases.yaml"))

RAG_CASE_IDS: list[str] = []
for case in RAG_CASES:
    RAG_CASE_IDS.append(case.id)


@pytest.mark.parametrize(
    "case",
    RAG_CASES,
    ids=RAG_CASE_IDS,
)
def test_rag_case(case: RagCase, workspace_slug: str) -> None:
    client = AnythingLLMClient()

    result = evaluate_rag_case(
        client=client,
        workspace_slug=workspace_slug,
        case=case,
    )

    assert result.passed, format_evaluation_failure(result)
