import uuid
from pathlib import Path

from clients.anythingllm_client import AnythingLLMClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STANDARD_DOCUMENTS: list[Path] = [
    PROJECT_ROOT / "fixtures" / "documents" / "acme_support_policy.md",
]

CHAT_PROVIDER = "ollama"
CHAT_MODEL = "qwen3:8b"


def prepare_workspace(
    client: AnythingLLMClient,
    documents: list[Path],
) -> str:
    unique_suffix = uuid.uuid4().hex[:8]
    workspace_name = f"rag-bench-{unique_suffix}"
    requested_slug = f"rag-bench-{unique_suffix}"

    workspace_slug = client.create_workspace(
        name=workspace_name,
        slug=requested_slug,
    )

    client.update_workspace(
        workspace_slug=workspace_slug,
        chat_provider=CHAT_PROVIDER,
        chat_model=CHAT_MODEL,
    )

    for document_path in documents:
        document_location = client.upload_document(document_path)
        client.add_document_to_workspace(
            workspace_slug=workspace_slug,
            document_location=document_location,
        )

    return workspace_slug


def cleanup_workspace(
    client: AnythingLLMClient,
    workspace_slug: str,
) -> None:
    client.delete_workspace(workspace_slug)
