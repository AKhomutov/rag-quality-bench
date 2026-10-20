from __future__ import annotations

import json

import pytest

from split_claim_results import parse_base_claim_judgement


def _payload(**overrides: object) -> str:
    data: dict[str, object] = {
        "claim_id": "C7",
        "status": "contradicted",
        "reference_fact_ids": ["F1", "F6"],
        "provenance_issue": True,
        "explanation": "Severity 0 is not defined.",
    }
    data.update(overrides)
    return json.dumps(data)


def test_parse_base_claim_happy_path() -> None:
    result = parse_base_claim_judgement(_payload(), expected_claim_id="C7")
    assert result.claim_id == "C7"
    assert result.status == "contradicted"
    assert result.reference_fact_ids == ["F1", "F6"]
    assert result.provenance_issue is True


def test_exact_status_enum() -> None:
    for status in ("supported", "contradicted", "not_in_evidence"):
        result = parse_base_claim_judgement(
            _payload(status=status),
            expected_claim_id="C7",
        )
        assert result.status == status


def test_derived_status_rejected_in_base() -> None:
    with pytest.raises(ValueError, match="status"):
        parse_base_claim_judgement(
            _payload(status="derived_from_unsupported_claim"),
            expected_claim_id="C7",
        )


def test_unknown_fact_ids_rejected() -> None:
    with pytest.raises(ValueError, match="unknown IDs"):
        parse_base_claim_judgement(
            _payload(reference_fact_ids=["F99"]),
            expected_claim_id="C7",
        )


def test_claim_id_must_match_requested() -> None:
    with pytest.raises(ValueError, match="claim_id mismatch"):
        parse_base_claim_judgement(_payload(), expected_claim_id="C3")


def test_provenance_must_be_bool() -> None:
    with pytest.raises(TypeError, match="bool"):
        parse_base_claim_judgement(
            _payload(provenance_issue="yes"),
            expected_claim_id="C7",
        )
