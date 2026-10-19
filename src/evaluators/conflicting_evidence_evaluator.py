import uuid
from pathlib import Path

from clients.anythingllm_client import AnythingLLMClient
from conflicting_cases import ConflictingEvidenceCase
from evaluation_results import EvaluationResult
from evaluators.evaluation_utils import contains_any_expected, normalize_answer_text
from workspace_setup import (
    PROJECT_ROOT,
    PreparedWorkspace,
    cleanup_workspace,
    prepare_workspace,
)

DOCUMENTS_DIR = PROJECT_ROOT / "fixtures" / "documents"


def resolve_case_documents(case: ConflictingEvidenceCase) -> list[Path]:
    resolved: list[Path] = []
    for filename in case.documents:
        document_path = DOCUMENTS_DIR / filename
        if not document_path.is_file():
            msg = f"case {case.id!r}: document not found: {document_path}"
            raise FileNotFoundError(msg)
        resolved.append(document_path)
    return resolved


def run_conflicting_evidence_case(
    client: AnythingLLMClient,
    case: ConflictingEvidenceCase,
) -> str:
    documents = resolve_case_documents(case)
    workspace: PreparedWorkspace | None = None

    try:
        workspace = prepare_workspace(client, documents)

        thread_slug = f"{case.id}-{uuid.uuid4().hex[:8]}"
        created_slug = client.create_thread(workspace.slug, thread_slug)

        final_response = None
        for step in case.steps:
            final_response = client.thread_chat(
                workspace_slug=workspace.slug,
                thread_slug=created_slug,
                message=step.message,
            )

        if final_response is None:
            msg = f"case {case.id!r} produced no response"
            raise RuntimeError(msg)

        return final_response.final_answer
    finally:
        if workspace is not None:
            cleanup_workspace(client, workspace)


def score_conflicting_evidence_answer(
    case: ConflictingEvidenceCase,
    final_answer: str,
) -> EvaluationResult:
    normalized_answer = normalize_answer_text(final_answer)
    failures: list[str] = []

    for expected in case.expected_final_contains_all:
        if normalize_answer_text(expected) not in normalized_answer:
            failures.append(f"missing expected string {expected!r}")

    if not contains_any_expected(normalized_answer, case.expected_final_contains_any):
        failures.append("final answer missing any accepted conflict acknowledgment")

    for banned in case.expected_final_excludes:
        if normalize_answer_text(banned) in normalized_answer:
            failures.append(f"forbidden string present: {banned!r}")

    return EvaluationResult(
        suite="conflicting_evidence",
        case_id=case.id,
        category=case.category,
        passed=len(failures) == 0,
        expected_behavior=case.expected_behavior,
        failure_description=case.failure_description,
        final_answer=normalized_answer,
        failures=failures,
        tools_used=[],
        trial_summary=None,
    )


def evaluate_conflicting_evidence_case(
    client: AnythingLLMClient,
    case: ConflictingEvidenceCase,
) -> EvaluationResult:
    final_answer = run_conflicting_evidence_case(client, case)
    return score_conflicting_evidence_answer(case, final_answer)
