from clients.anythingllm_client import AnythingLLMClient, Source
from evaluation_results import EvaluationResult
from evaluators.evaluation_utils import contains_any_expected, normalize_answer_text
from rag_cases import RagCase

BASELINE_EXPECTED_BEHAVIOR = (
    "Return an answer consistent with the authoritative workspace source "
    "and retrieve the expected supporting evidence."
)
BASELINE_FAILURE_DESCRIPTION = (
    "The RAG response or retrieved evidence does not match the deterministic case expectations."
)


def contains_source_text(sources: list[Source], expected_text: str) -> bool:
    for source in sources:
        if expected_text in source.text:
            return True
    return False


def evaluate_rag_case(
    client: AnythingLLMClient,
    workspace_slug: str,
    case: RagCase,
) -> EvaluationResult:
    response = client.chat(
        workspace_slug=workspace_slug,
        message=case.question,
    )

    normalized_answer = normalize_answer_text(response.final_answer)
    failures: list[str] = []

    if not contains_any_expected(normalized_answer, case.expected_answer_contains_any):
        failures.append("final answer missing any expected answer variant")

    if (
        case.expected_answer_excludes is not None
        and case.expected_answer_excludes in normalized_answer
    ):
        failures.append(f"forbidden answer string present: {case.expected_answer_excludes!r}")

    if case.expected_source_contains is not None and not contains_source_text(
        response.sources,
        case.expected_source_contains,
    ):
        failures.append(f"expected source text not retrieved: {case.expected_source_contains!r}")

    return EvaluationResult(
        suite="baseline",
        case_id=case.id,
        category=case.category,
        passed=len(failures) == 0,
        expected_behavior=BASELINE_EXPECTED_BEHAVIOR,
        failure_description=BASELINE_FAILURE_DESCRIPTION,
        final_answer=normalized_answer,
        failures=failures,
        tools_used=[],
        trial_summary=None,
    )
