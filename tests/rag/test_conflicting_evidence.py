from pathlib import Path

import pytest

from clients.anythingllm_client import AnythingLLMClient
from conflicting_cases import ConflictingEvidenceCase, load_conflicting_evidence_cases
from evaluation_assertions import format_evaluation_failure
from evaluators.conflicting_evidence_evaluator import evaluate_conflicting_evidence_case

CONFLICTING_CASES = load_conflicting_evidence_cases(
    Path("fixtures/datasets/conflicting_evidence.yaml")
)

CONFLICTING_CASE_IDS: list[str] = []
for case in CONFLICTING_CASES:
    CONFLICTING_CASE_IDS.append(case.id)


@pytest.mark.parametrize(
    "case",
    CONFLICTING_CASES,
    ids=CONFLICTING_CASE_IDS,
)
def test_conflicting_evidence_case(case: ConflictingEvidenceCase) -> None:
    client = AnythingLLMClient()

    result = evaluate_conflicting_evidence_case(
        client=client,
        case=case,
    )

    assert result.passed, format_evaluation_failure(result)
