from __future__ import annotations

from claim_level_results import ALLOWED_COVERAGE_STATUSES, ALLOWED_SUPPORT_STATUSES
from json_types import JsonObject, JsonValue
from long_form_severity_facts import canonical_reference_fact_ids

CLAIM_LEVEL_SCHEMA_VERSION = 1
CLAIM_LEVEL_SCHEMA_NAME = "claim_level_judge_result"


def _support_status_enum() -> list[JsonValue]:
    values: list[JsonValue] = []
    for status in sorted(ALLOWED_SUPPORT_STATUSES):
        values.append(status)
    return values


def _coverage_status_enum() -> list[JsonValue]:
    values: list[JsonValue] = []
    for status in sorted(ALLOWED_COVERAGE_STATUSES):
        values.append(status)
    return values


def _reference_fact_id_enum() -> list[JsonValue]:
    values: list[JsonValue] = []
    for fact_id in canonical_reference_fact_ids():
        values.append(fact_id)
    return values


def build_claim_level_json_schema() -> JsonObject:
    """JSON Schema matching the strict claim-level parser contract."""
    claim_item: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "claim",
            "candidate_excerpt",
            "support_status",
            "reference_fact_ids",
            "provenance_issue",
            "explanation",
        ],
        "properties": {
            "claim": {"type": "string", "minLength": 1},
            "candidate_excerpt": {"type": "string", "minLength": 1},
            "support_status": {
                "type": "string",
                "enum": _support_status_enum(),
            },
            "reference_fact_ids": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": _reference_fact_id_enum(),
                },
            },
            "provenance_issue": {"type": "boolean"},
            "explanation": {"type": "string", "minLength": 1},
        },
    }

    coverage_item: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": ["question_number", "status", "explanation"],
        "properties": {
            "question_number": {
                "type": "integer",
                "minimum": 1,
                "maximum": 40,
            },
            "status": {
                "type": "string",
                "enum": _coverage_status_enum(),
            },
            "explanation": {"type": "string", "minLength": 1},
        },
    }

    schema: JsonObject = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "claims",
            "question_coverage",
            "final_table_present",
            "final_summary_present",
            "rationale",
        ],
        "properties": {
            "claims": {
                "type": "array",
                "minItems": 1,
                "items": claim_item,
            },
            "question_coverage": {
                "type": "array",
                "items": coverage_item,
            },
            "final_table_present": {"type": "boolean"},
            "final_summary_present": {"type": "boolean"},
            "rationale": {"type": "string", "minLength": 1},
        },
    }
    return schema
