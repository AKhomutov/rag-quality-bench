from dataclasses import dataclass
from pathlib import Path

from agent_cases import AgentToolSelectionCase
from clients.anythingllm_client import AnythingLLMClient
from evaluation_results import EvaluationResult, TrialSummary
from evaluators.evaluation_utils import contains_any_expected, normalize_answer_text
from workspace_setup import (
    PROJECT_ROOT,
    PreparedWorkspace,
    cleanup_workspace,
    prepare_workspace,
)

DOCUMENTS_DIR = PROJECT_ROOT / "fixtures" / "documents"


@dataclass(frozen=True)
class AgentTrialResult:
    passed: bool
    final_answer: str
    failures: list[str]
    tools_used: list[str]


def resolve_case_documents(case: AgentToolSelectionCase) -> list[Path]:
    resolved: list[Path] = []
    for filename in case.documents:
        document_path = DOCUMENTS_DIR / filename
        if not document_path.is_file():
            msg = f"case {case.id!r}: document not found: {document_path}"
            raise FileNotFoundError(msg)
        resolved.append(document_path)
    return resolved


def tool_was_used(tools_used: list[str], tool_name: str) -> bool:
    for used in tools_used:
        if used == tool_name:
            return True
    return False


def any_preferred_tool_used(tools_used: list[str], preferred_tools: list[str]) -> bool:
    for preferred in preferred_tools:
        if tool_was_used(tools_used, preferred):
            return True
    return False


def unique_tools(tools_used: list[str]) -> list[str]:
    unique: list[str] = []
    for tool_name in tools_used:
        if tool_name not in unique:
            unique.append(tool_name)
    return unique


def run_agent_trial(
    client: AnythingLLMClient,
    documents: list[Path],
    case: AgentToolSelectionCase,
) -> AgentTrialResult:
    workspace: PreparedWorkspace | None = None

    try:
        workspace = prepare_workspace(client, documents)
        response = client.agent_chat(
            workspace_slug=workspace.slug,
            message=case.question,
        )

        normalized_answer = normalize_answer_text(response.final_answer)
        tools_used: list[str] = []
        for tool_call in response.tool_calls:
            tools_used.append(tool_call.name)

        failures: list[str] = []

        if not contains_any_expected(normalized_answer, case.expected_answer_contains_any):
            failures.append("final answer missing any expected answer variant")

        for forbidden in case.forbidden_tools:
            if tool_was_used(tools_used, forbidden):
                failures.append(f"forbidden tool called: {forbidden!r}")

        if case.preferred_tools and not any_preferred_tool_used(
            tools_used,
            case.preferred_tools,
        ):
            failures.append("expected workspace/RAG tool was not used")

        return AgentTrialResult(
            passed=len(failures) == 0,
            final_answer=normalized_answer,
            failures=failures,
            tools_used=tools_used,
        )
    finally:
        if workspace is not None:
            cleanup_workspace(client, workspace)


def evaluate_agent_tool_selection_case(
    client: AnythingLLMClient,
    case: AgentToolSelectionCase,
) -> EvaluationResult:
    documents = resolve_case_documents(case)

    trial_results: list[AgentTrialResult] = []
    for _trial_number in range(case.trials):
        trial_results.append(
            run_agent_trial(
                client=client,
                documents=documents,
                case=case,
            )
        )

    total_trials = len(trial_results)
    passed_trials = 0
    for trial in trial_results:
        if trial.passed:
            passed_trials += 1
    failed_trials = total_trials - passed_trials
    pass_rate = passed_trials / total_trials

    tool_usage: dict[str, int] = {}
    tools_seen_order: list[str] = []
    for trial in trial_results:
        for tool_name in unique_tools(trial.tools_used):
            if tool_name not in tool_usage:
                tool_usage[tool_name] = 0
                tools_seen_order.append(tool_name)
            tool_usage[tool_name] = tool_usage[tool_name] + 1

    case_passed = pass_rate >= case.minimum_pass_rate

    failures: list[str] = []
    if not case_passed:
        failures.append(
            f"pass rate {pass_rate:.2f} is below required minimum {case.minimum_pass_rate:.2f}"
        )
        failures.append(f"{failed_trials}/{total_trials} trials failed")

        trial_number = 1
        for trial in trial_results:
            if not trial.passed:
                for failure in trial.failures:
                    failures.append(f"trial {trial_number}: {failure}")
            trial_number += 1

    trial_summary = TrialSummary(
        total=total_trials,
        passed=passed_trials,
        failed=failed_trials,
        pass_rate=pass_rate,
        minimum_pass_rate=case.minimum_pass_rate,
        tool_usage=tool_usage,
    )

    return EvaluationResult(
        suite="agent_tool_selection",
        case_id=case.id,
        category=case.category,
        passed=case_passed,
        expected_behavior=case.expected_behavior,
        failure_description=case.failure_description,
        final_answer=(f"Repeated agent evaluation: {passed_trials}/{total_trials} trials passed."),
        failures=failures,
        tools_used=tools_seen_order,
        trial_summary=trial_summary,
    )
