from collections.abc import Generator

import pytest

from clients.anythingllm_client import AnythingLLMClient
from workspace_setup import STANDARD_DOCUMENTS, cleanup_workspace, prepare_workspace


@pytest.fixture(scope="session")
def workspace_slug() -> Generator[str, None, None]:
    client = AnythingLLMClient()
    workspace = prepare_workspace(
        client,
        STANDARD_DOCUMENTS,
    )
    try:
        yield workspace.slug
    finally:
        cleanup_workspace(client, workspace)
