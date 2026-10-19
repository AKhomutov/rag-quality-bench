from __future__ import annotations

from claim_level_diagnostics import classify_parse_error


def test_truncated_json_classified_as_invalid_json_before_enum_inspection() -> None:
    error = (
        "claim-level judge execution error for 'x' trial=1: "
        "judge output is not valid JSON: Unterminated string starting at: "
        "line 160 column 7 (char 8150)\n"
        "raw response:\n"
        '{\n  "support_status": "supported",\n  "provenance_issue'
    )
    assert classify_parse_error(error) == "invalid_json"


def test_support_status_error_after_valid_json_path() -> None:
    error = (
        "claim-level judge execution error for 'x' trial=1: "
        "claims[0].support_status must be one of "
        "['contradicted', 'derived_from_unsupported_claim', "
        "'not_in_evidence', 'supported'], got 'not in evidence'\n"
        "raw response:\n{}"
    )
    assert classify_parse_error(error) == "invalid_support_status"


def test_unknown_fact_id_classification() -> None:
    error = (
        "claim-level judge execution error for 'x' trial=1: "
        "claims[12].reference_fact_ids contains unknown IDs: ['F19']\n"
        "raw response:\n{}"
    )
    assert classify_parse_error(error) == "unknown_fact_id"
