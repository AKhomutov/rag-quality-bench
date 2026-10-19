from __future__ import annotations

import json

import pytest

from claim_level_results import (
    ClaimAuditItem,
    ClaimLevelJudgeResult,
    QuestionCoverageItem,
    claim_level_passed,
    parse_claim_level_judge_result,
)


def _coverage(status: str = "answered") -> list[dict[str, object]]:
    items: list[dict[str, object]] = []
    for number in range(1, 41):
        items.append(
            {
                "question_number": number,
                "status": status,
                "explanation": f"q{number}",
            }
        )
    return items


def _claim(
    *,
    support_status: str = "supported",
    provenance_issue: bool = False,
    fact_ids: list[str] | None = None,
) -> dict[str, object]:
    return {
        "claim": "Acme defines Severity 1.",
        "candidate_excerpt": "Severity 1 — Critical",
        "support_status": support_status,
        "reference_fact_ids": ["F2"] if fact_ids is None else fact_ids,
        "provenance_issue": provenance_issue,
        "explanation": "matches F2",
    }


def _payload(**overrides: object) -> str:
    data: dict[str, object] = {
        "claims": [_claim()],
        "question_coverage": _coverage(),
        "final_table_present": True,
        "final_summary_present": True,
        "rationale": "all good",
    }
    data.update(overrides)
    return json.dumps(data)


def test_parse_claim_audit_item_happy_path() -> None:
    result = parse_claim_level_judge_result(_payload())
    assert len(result.claims) == 1
    item = result.claims[0]
    assert isinstance(item, ClaimAuditItem)
    assert item.support_status == "supported"
    assert item.provenance_issue is False
    assert item.reference_fact_ids == ["F2"]


def test_valid_support_statuses() -> None:
    for status in (
        "supported",
        "contradicted",
        "not_in_evidence",
        "derived_from_unsupported_claim",
    ):
        result = parse_claim_level_judge_result(_payload(claims=[_claim(support_status=status)]))
        assert result.claims[0].support_status == status


def test_invalid_support_status_rejected() -> None:
    with pytest.raises(ValueError, match="support_status"):
        parse_claim_level_judge_result(_payload(claims=[_claim(support_status="maybe")]))


def test_provenance_must_be_bool() -> None:
    claim = _claim()
    claim["provenance_issue"] = "true"
    with pytest.raises(TypeError, match="provenance_issue"):
        parse_claim_level_judge_result(_payload(claims=[claim]))


def test_valid_reference_fact_ids() -> None:
    result = parse_claim_level_judge_result(_payload(claims=[_claim(fact_ids=["F1", "F6"])]))
    assert result.claims[0].reference_fact_ids == ["F1", "F6"]


def test_unknown_fact_id_rejected() -> None:
    with pytest.raises(ValueError, match="unknown IDs"):
        parse_claim_level_judge_result(_payload(claims=[_claim(fact_ids=["F99"])]))


def test_exact_1_to_40_coverage() -> None:
    result = parse_claim_level_judge_result(_payload())
    assert [item.question_number for item in result.question_coverage] == list(range(1, 41))
    assert all(isinstance(item, QuestionCoverageItem) for item in result.question_coverage)


def test_duplicate_question_numbers_rejected() -> None:
    coverage = _coverage()
    coverage[5]["question_number"] = 1
    with pytest.raises(ValueError, match="duplicates"):
        parse_claim_level_judge_result(_payload(question_coverage=coverage))


def test_missing_question_number_rejected() -> None:
    coverage = _coverage()[:-1]
    with pytest.raises(ValueError, match="missing"):
        parse_claim_level_judge_result(_payload(question_coverage=coverage))


def test_invalid_coverage_status_rejected() -> None:
    coverage = _coverage()
    coverage[0]["status"] = "skipped"
    with pytest.raises(ValueError, match="status"):
        parse_claim_level_judge_result(_payload(question_coverage=coverage))


def test_final_table_and_summary_booleans() -> None:
    result = parse_claim_level_judge_result(
        _payload(final_table_present=False, final_summary_present=True)
    )
    assert result.final_table_present is False
    assert result.final_summary_present is True

    bad_payload = (
        '{"claims":[{"claim":"c","candidate_excerpt":"e","support_status":"supported",'
        '"reference_fact_ids":["F1"],"provenance_issue":false,"explanation":"x"}],'
        '"question_coverage":'
        + json.dumps(_coverage())
        + ',"final_table_present":1,"final_summary_present":true,"rationale":"r"}'
    )
    with pytest.raises(TypeError, match="final_table_present"):
        parse_claim_level_judge_result(bad_payload)


def _result_from(
    *,
    support_status: str = "supported",
    provenance_issue: bool = False,
    coverage_status: str = "answered",
    final_table: bool = True,
    final_summary: bool = True,
) -> ClaimLevelJudgeResult:
    return parse_claim_level_judge_result(
        _payload(
            claims=[_claim(support_status=support_status, provenance_issue=provenance_issue)],
            question_coverage=_coverage(coverage_status),
            final_table_present=final_table,
            final_summary_present=final_summary,
        )
    )


def test_verdict_all_supported_pass() -> None:
    assert claim_level_passed(_result_from()) is True


def test_verdict_one_contradicted_fail() -> None:
    assert claim_level_passed(_result_from(support_status="contradicted")) is False


def test_verdict_one_not_in_evidence_fail() -> None:
    assert claim_level_passed(_result_from(support_status="not_in_evidence")) is False


def test_verdict_one_derived_fail() -> None:
    assert (
        claim_level_passed(_result_from(support_status="derived_from_unsupported_claim")) is False
    )


def test_verdict_provenance_issue_fail() -> None:
    assert claim_level_passed(_result_from(provenance_issue=True)) is False


def test_verdict_missing_question_fail() -> None:
    assert claim_level_passed(_result_from(coverage_status="missing")) is False


def test_verdict_partial_coverage_does_not_fail() -> None:
    assert claim_level_passed(_result_from(coverage_status="partial")) is True


def test_verdict_missing_table_or_summary_fail() -> None:
    assert claim_level_passed(_result_from(final_table=False)) is False
    assert claim_level_passed(_result_from(final_summary=False)) is False
