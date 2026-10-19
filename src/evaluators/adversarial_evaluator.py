import uuid

from adversarial_cases import AdversarialCase
from clients.anythingllm_client import AnythingLLMClient


def run_adversarial_case(
    client: AnythingLLMClient,
    workspace_slug: str,
    case: AdversarialCase,
) -> str:
    thread_slug = f"{case.id}-{uuid.uuid4().hex[:8]}"
    created_slug = client.create_thread(workspace_slug, thread_slug)

    final_response = None
    for step in case.steps:
        final_response = client.thread_chat(
            workspace_slug=workspace_slug,
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
