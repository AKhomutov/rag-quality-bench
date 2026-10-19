from pathlib import Path

import pytest

from adversarial_cases import AdversarialCase, load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient
from evaluators.adversarial_evaluator import run_adversarial_case

ADVERSARIAL_CASES = load_adversarial_cases(Path("fixtures/datasets/adversarial_conversations.yaml"))

ADVERSARIAL_CASE_IDS: list[str] = []
for case in ADVERSARIAL_CASES:
    ADVERSARIAL_CASE_IDS.append(case.id)


@pytest.mark.parametrize(
    "case",
    ADVERSARIAL_CASES,
    ids=ADVERSARIAL_CASE_IDS,
)
def test_adversarial_conversation_executes(case: AdversarialCase, workspace_slug: str) -> None:
    """Execution smoke test; content correctness is owned by the semantic judge."""
    client = AnythingLLMClient()

    final_answer = run_adversarial_case(
        client=client,
        workspace_slug=workspace_slug,
        case=case,
    )

    assert final_answer.strip() != ""
