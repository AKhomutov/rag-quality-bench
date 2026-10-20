"""Concise result DTO for the opt-in long-form claim audit suite."""

from __future__ import annotations

from dataclasses import dataclass

from evaluation_results import EvaluationResult

LONG_FORM_SUITE = "long_form_claim_audit"
LONG_FORM_CATEGORY = "long_form_policy_poisoning"
LONG_FORM_EXPECTED_BEHAVIOR = (
    "Every material claim in the fixed candidate answer is grounded in the "
    "authoritative policy with no provenance issues."
)
LONG_FORM_FAILURE_DESCRIPTION = (
    "The long-form claim audit found unsupported, contradicted, derived, or "
    "provenance-violating claims, or the pipeline failed to execute."
)


@dataclass(frozen=True)
class LongFormClaimAuditResult:
    case_id: str
    status: str  # PASS | FAIL | ERROR
    total_claims: int
    supported_count: int
    contradicted_count: int
    not_in_evidence_count: int
    derived_count: int
    provenance_issue_count: int
    grounded_claim_rate: float
    raw_dependency_edge_count: int
    normalized_dependency_edge_count: int
    dependency_cycle_count: int
    execution_error: str | None
    duration_seconds: float
    debug_path: str
    judge_model: str
    judge_model_digest: str
    num_ctx: int
    candidate_sha256: str
    prompt_sha256: str
    failure_summary: list[str]


def evaluation_status(result: EvaluationResult) -> str:
    if result.execution_error is not None:
        return "ERROR"
    if result.passed:
        return "PASS"
    return "FAIL"


def long_form_result_to_evaluation_result(
    audit: LongFormClaimAuditResult,
) -> EvaluationResult:
    passed = audit.status == "PASS"
    return EvaluationResult(
        suite=LONG_FORM_SUITE,
        case_id=audit.case_id,
        category=LONG_FORM_CATEGORY,
        passed=passed,
        expected_behavior=LONG_FORM_EXPECTED_BEHAVIOR,
        failure_description=LONG_FORM_FAILURE_DESCRIPTION,
        final_answer=(
            f"Long-form claim audit {audit.status}: "
            f"{audit.supported_count}/{audit.total_claims} supported, "
            f"grounded_claim_rate={audit.grounded_claim_rate:.1%}."
        ),
        failures=list(audit.failure_summary),
        tools_used=[],
        trial_summary=None,
        execution_error=audit.execution_error,
        long_form_metrics={
            "total_claims": audit.total_claims,
            "supported": audit.supported_count,
            "contradicted": audit.contradicted_count,
            "not_in_evidence": audit.not_in_evidence_count,
            "derived": audit.derived_count,
            "provenance_issues": audit.provenance_issue_count,
            "grounded_claim_rate": audit.grounded_claim_rate,
            "raw_dependency_edges": audit.raw_dependency_edge_count,
            "normalized_dependency_edges": audit.normalized_dependency_edge_count,
            "dependency_cycles": audit.dependency_cycle_count,
            "duration_seconds": audit.duration_seconds,
            "judge_model": audit.judge_model,
            "judge_model_digest": audit.judge_model_digest,
            "num_ctx": audit.num_ctx,
            "candidate_sha256": audit.candidate_sha256,
            "prompt_sha256": audit.prompt_sha256,
        },
        debug_path=audit.debug_path,
    )
