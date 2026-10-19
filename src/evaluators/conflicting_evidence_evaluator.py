import uuid
from pathlib import Path

from clients.anythingllm_client import AnythingLLMClient
from conflicting_cases import ConflictingEvidenceCase
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

        final_answer = final_response.final_answer.strip()
        if final_answer == "":
            msg = f"case {case.id!r} produced an empty final answer"
            raise RuntimeError(msg)

        return final_answer
    finally:
        if workspace is not None:
            cleanup_workspace(client, workspace)
