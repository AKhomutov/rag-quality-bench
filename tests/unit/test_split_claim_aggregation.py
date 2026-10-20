from __future__ import annotations

from split_claim_results import (
    BaseClaimJudgement,
    DependencyJudgement,
    ExtractedClaim,
    FinalClaimJudgement,
    build_final_judgements,
    compute_split_claim_metrics,
    split_claim_passed,
)


def _final(
    *,
    claim_id: str = "C1",
    status: str = "supported",
    provenance_issue: bool = False,
    depends_on: list[str] | None = None,
) -> FinalClaimJudgement:
    return FinalClaimJudgement(
        claim_id=claim_id,
        claim=f"claim {claim_id}",
        candidate_excerpt="excerpt",
        status=status,
        reference_fact_ids=["F1"],
        provenance_issue=provenance_issue,
        depends_on=[] if depends_on is None else depends_on,
        explanation="ok",
    )


def test_all_supported_passes() -> None:
    finals = [_final(claim_id="C1"), _final(claim_id="C2")]
    assert split_claim_passed(finals) is True


def test_one_contradicted_fails() -> None:
    finals = [_final(claim_id="C1"), _final(claim_id="C2", status="contradicted")]
    assert split_claim_passed(finals) is False


def test_one_not_in_evidence_fails() -> None:
    finals = [_final(status="not_in_evidence")]
    assert split_claim_passed(finals) is False


def test_one_derived_fails() -> None:
    finals = [_final(status="derived_from_unsupported_claim", depends_on=["C9"])]
    assert split_claim_passed(finals) is False


def test_provenance_issue_fails() -> None:
    finals = [_final(provenance_issue=True)]
    assert split_claim_passed(finals) is False


def test_grounded_claim_rate() -> None:
    finals = [
        _final(claim_id="C1", status="supported"),
        _final(claim_id="C2", status="supported"),
        _final(claim_id="C3", status="contradicted"),
        _final(claim_id="C4", status="not_in_evidence"),
    ]
    metrics = compute_split_claim_metrics(finals)
    assert metrics.total_claims == 4
    assert metrics.supported == 2
    assert metrics.contradicted == 1
    assert metrics.not_in_evidence == 1
    assert metrics.derived_from_unsupported_claim == 0
    assert metrics.provenance_issues == 0
    assert metrics.grounded_claim_rate == 0.5


def test_build_final_applies_dependency_only_to_not_in_evidence() -> None:
    extracted = [
        ExtractedClaim("C1", "Severity 0 exists", "excerpt0"),
        ExtractedClaim("C2", "Severity 1 is critical", "excerpt1"),
        ExtractedClaim("C3", "single-org distinction", "excerpt3"),
    ]
    base = {
        "C1": BaseClaimJudgement("C1", "contradicted", ["F6"], True, "fake level"),
        "C2": BaseClaimJudgement("C2", "supported", ["F2"], False, "ok"),
        "C3": BaseClaimJudgement("C3", "not_in_evidence", [], True, "unsupported"),
    }
    deps = {
        "C3": DependencyJudgement(
            "C3",
            True,
            ["C1"],
            "derived from unsupported Severity 0",
        ),
    }
    finals = build_final_judgements(extracted, base, deps)
    by_id = {item.claim_id: item for item in finals}
    assert by_id["C1"].status == "contradicted"
    assert by_id["C1"].depends_on == []
    assert by_id["C2"].status == "supported"
    assert by_id["C3"].status == "derived_from_unsupported_claim"
    assert by_id["C3"].depends_on == ["C1"]


def test_build_final_keeps_not_in_evidence_when_not_derived() -> None:
    extracted = [ExtractedClaim("C1", "unknown detail", "excerpt")]
    base = {
        "C1": BaseClaimJudgement("C1", "not_in_evidence", [], False, "no evidence"),
    }
    deps = {
        "C1": DependencyJudgement("C1", False, [], "independent unsupported claim"),
    }
    finals = build_final_judgements(extracted, base, deps)
    assert finals[0].status == "not_in_evidence"
    assert finals[0].depends_on == []
