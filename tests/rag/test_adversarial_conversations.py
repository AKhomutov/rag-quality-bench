from pathlib import Path

import pytest

from adversarial_cases import AdversarialCase, load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient
from evaluation_assertions import format_evaluation_failure
from evaluators.adversarial_evaluator import evaluate_adversarial_case

ADVERSARIAL_CASES = load_adversarial_cases(Path("fixtures/datasets/adversarial_conversations.yaml"))

ADVERSARIAL_CASE_IDS: list[str] = []
for case in ADVERSARIAL_CASES:
    ADVERSARIAL_CASE_IDS.append(case.id)


@pytest.mark.parametrize(
    "case",
    ADVERSARIAL_CASES,
    ids=ADVERSARIAL_CASE_IDS,
)
def test_adversarial_conversation(case: AdversarialCase, workspace_slug: str) -> None:
    client = AnythingLLMClient()

    result = evaluate_adversarial_case(
        client=client,
        workspace_slug=workspace_slug,
        case=case,
    )

    assert result.passed, format_evaluation_failure(result)
