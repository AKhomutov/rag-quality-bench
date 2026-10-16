from pathlib import Path

import pytest

from agent_cases import AgentToolSelectionCase, load_agent_tool_selection_cases
from clients.anythingllm_client import AnythingLLMClient
from evaluation_assertions import format_evaluation_failure
from evaluators.agent_tool_selection_evaluator import evaluate_agent_tool_selection_case

AGENT_CASES = load_agent_tool_selection_cases(Path("fixtures/datasets/agent_tool_selection.yaml"))

AGENT_CASE_IDS: list[str] = []
for case in AGENT_CASES:
    AGENT_CASE_IDS.append(case.id)


@pytest.mark.parametrize(
    "case",
    AGENT_CASES,
    ids=AGENT_CASE_IDS,
)
def test_agent_tool_selection_case(case: AgentToolSelectionCase) -> None:
    client = AnythingLLMClient()

    result = evaluate_agent_tool_selection_case(
        client=client,
        case=case,
    )

    assert result.passed, format_evaluation_failure(result)
