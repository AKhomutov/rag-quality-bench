import os
from pathlib import Path

import pytest

from adversarial_cases import AdversarialCase, load_adversarial_cases
from clients.anythingllm_client import AnythingLLMClient
from evaluation_results import EvaluationResult
from evaluators.adversarial_evaluator import evaluate_adversarial_case

ADVERSARIAL_CASES = load_adversarial_cases(Path("fixtures/datasets/adversarial_conversations.yaml"))

ADVERSARIAL_CASE_IDS: list[str] = []
for case in ADVERSARIAL_CASES:
    ADVERSARIAL_CASE_IDS.append(case.id)


def format_evaluation_failure(result: EvaluationResult) -> str:
    failure_lines: list[str] = []
    for failure in result.failures:
        failure_lines.append(f"- {failure}")
    failures_text = "\n".join(failure_lines)

    return (
        f"case: {result.case_id}\n"
        f"category: {result.category}\n"
        f"expected behavior: {result.expected_behavior}\n"
        f"failure: {result.failure_description}\n"
        f"observed failures:\n{failures_text}\n"
        f"final answer:\n{result.final_answer}"
    )


@pytest.mark.parametrize(
    "case",
    ADVERSARIAL_CASES,
    ids=ADVERSARIAL_CASE_IDS,
)
def test_adversarial_conversation(case: AdversarialCase) -> None:
    client = AnythingLLMClient()
    workspace_slug = os.environ["ANYTHINGLLM_WORKSPACE"]

    result = evaluate_adversarial_case(
        client=client,
        workspace_slug=workspace_slug,
        case=case,
    )

    assert result.passed, format_evaluation_failure(result)
