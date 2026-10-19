"""Shared diagnostics helpers for claim-level judge runs."""

from __future__ import annotations


def classify_parse_error(error: str | None) -> str | None:
    """Classify parser failures using the exception head, not the embedded raw body.

    RuntimeError messages from claim-level tracing append the raw response after a
    marker. Classification must ignore that body so truncated JSON containing the
    substring ``support_status`` is not mislabeled.
    """
    if error is None:
        return None

    marker = "\nraw response:\n"
    head = error.split(marker, 1)[0]

    if "not valid JSON" in head:
        return "invalid_json"
    if "expected JSON object" in head:
        return "invalid_json"
    if "support_status" in head:
        return "invalid_support_status"
    if "unknown IDs" in head:
        return "unknown_fact_id"
    if "question_number" in head and "must be an int" in head:
        return "invalid_question_number"
    if "question_coverage must contain exactly" in head:
        return "duplicate_or_missing_coverage"
    if "status must be one of" in head and "question_coverage" in head:
        return "invalid_coverage_status"
    if "final_table_present" in head or "final_summary_present" in head:
        return "invalid_final_flags"
    if "claims" in head and ("non-empty" in head or "expected list" in head):
        return "invalid_claims"
    return "other_parse_error"
