import sys
import uuid
from dataclasses import dataclass
from pathlib import Path

from clients.anythingllm_client import AnythingLLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANDARD_DOCUMENTS: list[Path] = [
    PROJECT_ROOT / "fixtures" / "documents" / "acme_support_policy.md",
]

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "qwen3:8b"


@dataclass(frozen=True)
class PreparedWorkspace:
    slug: str
    document_locations: tuple[str, ...]


def prepare_workspace(
    client: AnythingLLMClient,
    documents: list[Path],
) -> PreparedWorkspace:
    unique_suffix = uuid.uuid4().hex[:8]
    workspace_name = f"rag-bench-{unique_suffix}"
    requested_slug = f"rag-bench-{unique_suffix}"

    workspace_slug = client.create_workspace(
        name=workspace_name,
        slug=requested_slug,
    )
    document_locations: list[str] = []

    try:
        client.update_workspace(
            workspace_slug=workspace_slug,
            chat_provider=CHAT_PROVIDER,
            chat_model=CHAT_MODEL,
        )

        for document_path in documents:
            document_location = client.upload_document(document_path)
            document_locations.append(document_location)
            client.add_document_to_workspace(
                workspace_slug=workspace_slug,
                document_location=document_location,
            )

        return PreparedWorkspace(
            slug=workspace_slug,
            document_locations=tuple(document_locations),
        )
    except Exception:
        _best_effort_cleanup_partial_setup(
            client=client,
            workspace_slug=workspace_slug,
            document_locations=document_locations,
        )
        raise


def _best_effort_cleanup_partial_setup(
    client: AnythingLLMClient,
    workspace_slug: str,
    document_locations: list[str],
) -> None:
    try:
        client.delete_workspace(workspace_slug)
    except Exception as cleanup_error:
        print(
            f"WARNING: failed to delete workspace {workspace_slug!r} after setup "
            f"failure: {cleanup_error}",
            file=sys.stderr,
        )

    if document_locations:
        try:
            client.delete_documents(document_locations)
        except Exception as cleanup_error:
            print(
                "WARNING: failed to delete uploaded documents after setup "
                f"failure: {cleanup_error}",
                file=sys.stderr,
            )


def cleanup_workspace(
    client: AnythingLLMClient,
    workspace: PreparedWorkspace,
) -> None:
    errors: list[str] = []

    try:
        client.delete_workspace(workspace.slug)
    except Exception as cleanup_error:
        message = f"failed to delete workspace {workspace.slug!r}: {cleanup_error}"
        print(f"WARNING: {message}", file=sys.stderr)
        errors.append(message)

    if workspace.document_locations:
        try:
            client.delete_documents(list(workspace.document_locations))
        except Exception as cleanup_error:
            message = (
                "failed to delete uploaded documents "
                f"{list(workspace.document_locations)!r}: {cleanup_error}"
            )
            print(f"WARNING: {message}", file=sys.stderr)
            errors.append(message)

    if errors:
        msg = "; ".join(errors)
        raise RuntimeError(msg)
