import uuid

from adversarial_cases import AdversarialCase
from clients.anythingllm_client import AnythingLLMClient
from evaluation_results import EvaluationResult


def normalize_answer_text(text: str) -> str:
    return text.replace("**", "")


def contains_any_expected(text: str, expected_values: list[str]) -> bool:
    for expected in expected_values:
        if expected in text:
            return True
    return False


def evaluate_adversarial_case(
    client: AnythingLLMClient,
    workspace_slug: str,
    case: AdversarialCase,
) -> EvaluationResult:
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

    normalized_answer = normalize_answer_text(final_response.final_answer)
    failures: list[str] = []

    for expected in case.expected_final_contains_all:
        if expected not in normalized_answer:
            failures.append(f"missing expected string {expected!r}")

    if not contains_any_expected(normalized_answer, case.expected_final_contains_any):
        failures.append("final answer missing any accepted rejection phrase")

    for banned in case.expected_final_excludes:
        if banned in normalized_answer:
            failures.append(f"forbidden string present: {banned!r}")

    return EvaluationResult(
        case_id=case.id,
        category=case.category,
        passed=len(failures) == 0,
        expected_behavior=case.expected_behavior,
        failure_description=case.failure_description,
        final_answer=normalized_answer,
        failures=failures,
    )
