from __future__ import annotations

import json

import pytest

from split_claim_results import (
    canonicalize_extracted_claim_ids,
    parse_extracted_claims,
)


def _payload(claims: list[dict[str, object]]) -> str:
    payload: dict[str, object] = {"claims": claims}
    return json.dumps(payload)


def test_parse_extracted_claims_happy_path() -> None:
    text = _payload(
        [
            {
                "claim_id": "C1",
                "claim": "Severity 1 is critical.",
                "candidate_excerpt": "Severity 1 — Critical",
            },
            {
                "claim_id": "C9",
                "claim": "Severity 0 exists.",
                "candidate_excerpt": "Severity 0 requires",
            },
        ]
    )
    claims = parse_extracted_claims(text)
    assert len(claims) == 2
    assert claims[0].claim_id == "C1"
    assert claims[1].claim_id == "C9"


def test_empty_claims_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_extracted_claims(_payload([]))


def test_duplicate_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        parse_extracted_claims(
            _payload(
                [
                    {
                        "claim_id": "C1",
                        "claim": "a",
                        "candidate_excerpt": "a",
                    },
                    {
                        "claim_id": "C1",
                        "claim": "b",
                        "candidate_excerpt": "b",
                    },
                ]
            )
        )


def test_empty_claim_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_extracted_claims(
            _payload(
                [
                    {
                        "claim_id": "C1",
                        "claim": "   ",
                        "candidate_excerpt": "excerpt",
                    }
                ]
            )
        )


def test_empty_excerpt_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        parse_extracted_claims(
            _payload(
                [
                    {
                        "claim_id": "C1",
                        "claim": "claim",
                        "candidate_excerpt": "",
                    }
                ]
            )
        )


def test_invalid_claim_id_rejected() -> None:
    with pytest.raises(ValueError, match="C<number>"):
        parse_extracted_claims(
            _payload(
                [
                    {
                        "claim_id": "claim-1",
                        "claim": "claim",
                        "candidate_excerpt": "excerpt",
                    }
                ]
            )
        )


def test_canonicalize_ids_preserves_text() -> None:
    claims = parse_extracted_claims(
        _payload(
            [
                {
                    "claim_id": "C9",
                    "claim": "first text",
                    "candidate_excerpt": "first excerpt",
                },
                {
                    "claim_id": "C2",
                    "claim": "second text",
                    "candidate_excerpt": "second excerpt",
                },
            ]
        )
    )
    canonical = canonicalize_extracted_claim_ids(claims)
    assert [item.claim_id for item in canonical] == ["C1", "C2"]
    assert canonical[0].claim == "first text"
    assert canonical[1].candidate_excerpt == "second excerpt"
