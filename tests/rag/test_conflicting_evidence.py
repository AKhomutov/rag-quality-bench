from pathlib import Path

import pytest

from clients.anythingllm_client import AnythingLLMClient
from conflicting_cases import ConflictingEvidenceCase, load_conflicting_evidence_cases
from evaluators.conflicting_evidence_evaluator import run_conflicting_evidence_case

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
def test_conflicting_evidence_case_executes(case: ConflictingEvidenceCase) -> None:
    """Execution smoke test; content correctness is owned by the semantic judge."""
    client = AnythingLLMClient()

    final_answer = run_conflicting_evidence_case(
        client=client,
        case=case,
    )

    assert final_answer.strip() != ""
