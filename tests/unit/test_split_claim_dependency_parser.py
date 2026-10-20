from __future__ import annotations

import json

import pytest

from split_claim_results import parse_dependency_judgement

KNOWN = frozenset({"C1", "C2", "C3", "C12"})


def _payload(**overrides: object) -> str:
    data: dict[str, object] = {
        "claim_id": "C12",
        "derived": True,
        "depends_on": ["C3"],
        "explanation": "depends on unsupported Severity 0 premise",
    }
    data.update(overrides)
    return json.dumps(data)


def test_parse_dependency_happy_path() -> None:
    result = parse_dependency_judgement(
        _payload(),
        expected_claim_id="C12",
        known_claim_ids=KNOWN,
    )
    assert result.derived is True
    assert result.depends_on == ["C3"]


def test_derived_false_requires_empty_depends_on() -> None:
    result = parse_dependency_judgement(
        _payload(derived=False, depends_on=[]),
        expected_claim_id="C12",
        known_claim_ids=KNOWN,
    )
    assert result.derived is False
    assert result.depends_on == []


def test_derived_false_with_depends_on_rejected() -> None:
    with pytest.raises(ValueError, match="empty when derived=false"):
        parse_dependency_judgement(
            _payload(derived=False, depends_on=["C1"]),
            expected_claim_id="C12",
            known_claim_ids=KNOWN,
        )


def test_derived_true_requires_dependency() -> None:
    with pytest.raises(ValueError, match="non-empty when derived=true"):
        parse_dependency_judgement(
            _payload(derived=True, depends_on=[]),
            expected_claim_id="C12",
            known_claim_ids=KNOWN,
        )


def test_unknown_dependency_id_rejected() -> None:
    with pytest.raises(ValueError, match="unknown claim_id"):
        parse_dependency_judgement(
            _payload(depends_on=["C99"]),
            expected_claim_id="C12",
            known_claim_ids=KNOWN,
        )


def test_self_dependency_rejected() -> None:
    with pytest.raises(ValueError, match="must not include the current"):
        parse_dependency_judgement(
            _payload(depends_on=["C12"]),
            expected_claim_id="C12",
            known_claim_ids=KNOWN,
        )


def test_claim_id_must_match() -> None:
    with pytest.raises(ValueError, match="claim_id mismatch"):
        parse_dependency_judgement(
            _payload(),
            expected_claim_id="C1",
            known_claim_ids=KNOWN,
        )


def test_derived_must_be_bool() -> None:
    with pytest.raises(TypeError, match="bool"):
        parse_dependency_judgement(
            _payload(derived="yes"),
            expected_claim_id="C12",
            known_claim_ids=KNOWN,
        )
