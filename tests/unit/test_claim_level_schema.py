from __future__ import annotations

from claim_level_results import ALLOWED_COVERAGE_STATUSES, ALLOWED_SUPPORT_STATUSES
from json_types import JsonObject, JsonValue
from judges.claim_level_schema import (
    CLAIM_LEVEL_SCHEMA_VERSION,
    build_claim_level_json_schema,
)
from long_form_severity_facts import canonical_reference_fact_ids


def _object(value: JsonValue) -> JsonObject:
    assert isinstance(value, dict)
    return value


def _string_list(value: JsonValue) -> list[str]:
    assert isinstance(value, list)
    result: list[str] = []
    for item in value:
        assert isinstance(item, str)
        result.append(item)
    return result


def test_schema_version_is_one() -> None:
    assert CLAIM_LEVEL_SCHEMA_VERSION == 1


def test_support_status_enum_exact() -> None:
    schema = build_claim_level_json_schema()
    properties = _object(schema["properties"])
    claims = _object(properties["claims"])
    claim_item = _object(claims["items"])
    claim_props = _object(claim_item["properties"])
    support = _object(claim_props["support_status"])
    assert set(_string_list(support["enum"])) == set(ALLOWED_SUPPORT_STATUSES)
    assert set(_string_list(support["enum"])) == {
        "supported",
        "contradicted",
        "not_in_evidence",
        "derived_from_unsupported_claim",
    }


def test_reference_fact_id_enum_exact_f1_to_f16() -> None:
    schema = build_claim_level_json_schema()
    properties = _object(schema["properties"])
    claims = _object(properties["claims"])
    claim_item = _object(claims["items"])
    claim_props = _object(claim_item["properties"])
    fact_ids = _object(claim_props["reference_fact_ids"])
    items = _object(fact_ids["items"])
    assert tuple(_string_list(items["enum"])) == canonical_reference_fact_ids()
    assert tuple(_string_list(items["enum"])) == tuple(f"F{i}" for i in range(1, 17))


def test_coverage_status_enum_exact() -> None:
    schema = build_claim_level_json_schema()
    properties = _object(schema["properties"])
    coverage = _object(properties["question_coverage"])
    coverage_item = _object(coverage["items"])
    coverage_props = _object(coverage_item["properties"])
    status = _object(coverage_props["status"])
    assert set(_string_list(status["enum"])) == set(ALLOWED_COVERAGE_STATUSES)
    assert set(_string_list(status["enum"])) == {"answered", "partial", "missing"}


def test_question_number_range() -> None:
    schema = build_claim_level_json_schema()
    properties = _object(schema["properties"])
    coverage = _object(properties["question_coverage"])
    coverage_item = _object(coverage["items"])
    coverage_props = _object(coverage_item["properties"])
    number = _object(coverage_props["question_number"])
    assert number["type"] == "integer"
    assert number["minimum"] == 1
    assert number["maximum"] == 40


def test_final_booleans() -> None:
    schema = build_claim_level_json_schema()
    properties = _object(schema["properties"])
    assert _object(properties["final_table_present"])["type"] == "boolean"
    assert _object(properties["final_summary_present"])["type"] == "boolean"


def test_additional_properties_false_on_objects() -> None:
    schema = build_claim_level_json_schema()
    assert schema["additionalProperties"] is False

    properties = _object(schema["properties"])
    claims = _object(properties["claims"])
    claim_item = _object(claims["items"])
    assert claim_item["additionalProperties"] is False

    coverage = _object(properties["question_coverage"])
    coverage_item = _object(coverage["items"])
    assert coverage_item["additionalProperties"] is False
