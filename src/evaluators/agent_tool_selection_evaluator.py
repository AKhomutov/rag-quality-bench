from pathlib import Path

from agent_cases import AgentToolSelectionCase
from clients.anythingllm_client import AnythingLLMClient
from evaluation_results import EvaluationResult
from evaluators.evaluation_utils import contains_any_expected, normalize_answer_text
from workspace_setup import PROJECT_ROOT, cleanup_workspace, prepare_workspace

DOCUMENTS_DIR = PROJECT_ROOT / "fixtures" / "documents"


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


def evaluate_agent_tool_selection_case(
    client: AnythingLLMClient,
    case: AgentToolSelectionCase,
) -> EvaluationResult:
    documents = resolve_case_documents(case)
    workspace_slug: str | None = None

    try:
        workspace_slug = prepare_workspace(client, documents)
        response = client.agent_chat(
            workspace_slug=workspace_slug,
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

        return EvaluationResult(
            suite="agent_tool_selection",
            case_id=case.id,
            category=case.category,
            passed=len(failures) == 0,
            expected_behavior=case.expected_behavior,
            failure_description=case.failure_description,
            final_answer=normalized_answer,
            failures=failures,
            tools_used=tools_used,
        )
    finally:
        if workspace_slug is not None:
            cleanup_workspace(client, workspace_slug)
