from __future__ import annotations

import json

import pytest

from split_claim_results import (
    parse_base_claim_batch,
    parse_dependency_batch,
    partition_into_batches,
)


def _base_item(
    claim_id: str,
    *,
    status: str = "supported",
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "status": status,
        "reference_fact_ids": ["F1"],
        "provenance_issue": False,
        "explanation": f"ok {claim_id}",
    }


def _dep_item(
    claim_id: str,
    *,
    derived: bool = False,
    depends_on: list[str] | None = None,
) -> dict[str, object]:
    return {
        "claim_id": claim_id,
        "derived": derived,
        "depends_on": [] if depends_on is None else depends_on,
        "explanation": f"dep {claim_id}",
    }


def _base_payload(items: list[dict[str, object]]) -> str:
    payload: dict[str, object] = {"judgements": items}
    return json.dumps(payload)


def _dep_payload(items: list[dict[str, object]]) -> str:
    payload: dict[str, object] = {"dependencies": items}
    return json.dumps(payload)


def test_parse_base_batch_valid() -> None:
    text = _base_payload(
        [
            _base_item("C2", status="contradicted"),
            _base_item("C1"),
        ]
    )
    judgements = parse_base_claim_batch(text, expected_claim_ids=frozenset({"C1", "C2"}))
    by_id = {item.claim_id: item for item in judgements}
    assert by_id["C1"].status == "supported"
    assert by_id["C2"].status == "contradicted"


def test_base_batch_missing_judgement_rejected() -> None:
    text = _base_payload([_base_item("C1")])
    with pytest.raises(ValueError, match="missing"):
        parse_base_claim_batch(text, expected_claim_ids=frozenset({"C1", "C2"}))


def test_base_batch_duplicate_rejected() -> None:
    text = _base_payload([_base_item("C1"), _base_item("C1")])
    with pytest.raises(ValueError, match="duplicate"):
        parse_base_claim_batch(text, expected_claim_ids=frozenset({"C1"}))


def test_base_batch_unknown_id_rejected() -> None:
    text = _base_payload([_base_item("C1"), _base_item("C99")])
    with pytest.raises(ValueError, match="unknown"):
        parse_base_claim_batch(text, expected_claim_ids=frozenset({"C1"}))


def test_base_batch_requires_exact_once_coverage() -> None:
    text = _base_payload(
        [
            _base_item("C1"),
            _base_item("C2"),
            _base_item("C3"),
        ]
    )
    with pytest.raises(ValueError, match="coverage mismatch"):
        parse_base_claim_batch(text, expected_claim_ids=frozenset({"C1", "C2"}))


def test_parse_dependency_batch_valid() -> None:
    text = _dep_payload(
        [
            _dep_item("C10", derived=True, depends_on=["C1"]),
            _dep_item("C11"),
        ]
    )
    judgements = parse_dependency_batch(
        text,
        expected_claim_ids=frozenset({"C10", "C11"}),
        known_claim_ids=frozenset({"C1", "C10", "C11"}),
    )
    by_id = {item.claim_id: item for item in judgements}
    assert by_id["C10"].derived is True
    assert by_id["C10"].depends_on == ["C1"]
    assert by_id["C11"].derived is False


def test_dependency_batch_missing_rejected() -> None:
    text = _dep_payload([_dep_item("C10")])
    with pytest.raises(ValueError, match="missing"):
        parse_dependency_batch(
            text,
            expected_claim_ids=frozenset({"C10", "C11"}),
            known_claim_ids=frozenset({"C10", "C11"}),
        )


def test_dependency_batch_duplicate_rejected() -> None:
    text = _dep_payload([_dep_item("C10"), _dep_item("C10")])
    with pytest.raises(ValueError, match="duplicate"):
        parse_dependency_batch(
            text,
            expected_claim_ids=frozenset({"C10"}),
            known_claim_ids=frozenset({"C10"}),
        )


def test_dependency_batch_unknown_rejected() -> None:
    text = _dep_payload([_dep_item("C99")])
    with pytest.raises(ValueError, match="unknown"):
        parse_dependency_batch(
            text,
            expected_claim_ids=frozenset({"C10"}),
            known_claim_ids=frozenset({"C10", "C99"}),
        )


def test_partition_empty() -> None:
    assert partition_into_batches([], 6) == []


def test_partition_one() -> None:
    assert partition_into_batches(["a"], 6) == [["a"]]


def test_partition_exact_batch_size() -> None:
    items = list(range(6))
    assert partition_into_batches(items, 6) == [items]


def test_partition_batch_size_plus_one() -> None:
    items = list(range(7))
    assert partition_into_batches(items, 6) == [list(range(6)), [6]]


def test_partition_multiple_batches() -> None:
    items = list(range(13))
    assert partition_into_batches(items, 6) == [
        list(range(6)),
        list(range(6, 12)),
        [12],
    ]


def test_partition_rejects_invalid_size() -> None:
    with pytest.raises(ValueError, match="batch_size"):
        partition_into_batches([1], 0)
