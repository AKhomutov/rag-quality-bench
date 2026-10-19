from __future__ import annotations

import json
from dataclasses import dataclass
from typing import cast

from long_form_severity_facts import EXPECTED_QUESTION_NUMBERS, KNOWN_REFERENCE_FACT_IDS

ALLOWED_SUPPORT_STATUSES = frozenset(
    {
        "supported",
        "contradicted",
        "not_in_evidence",
        "derived_from_unsupported_claim",
    }
)

ALLOWED_COVERAGE_STATUSES = frozenset(
    {
        "answered",
        "partial",
        "missing",
    }
)

FAILING_SUPPORT_STATUSES = frozenset(
    {
        "contradicted",
        "not_in_evidence",
        "derived_from_unsupported_claim",
    }
)


@dataclass(frozen=True)
class ClaimAuditItem:
    claim: str
    candidate_excerpt: str
    support_status: str
    reference_fact_ids: list[str]
    provenance_issue: bool
    explanation: str


@dataclass(frozen=True)
class QuestionCoverageItem:
    question_number: int
    status: str
    explanation: str


@dataclass(frozen=True)
class ClaimLevelJudgeResult:
    claims: list[ClaimAuditItem]
    question_coverage: list[QuestionCoverageItem]
    final_table_present: bool
    final_summary_present: bool
    rationale: str


def claim_level_passed(result: ClaimLevelJudgeResult) -> bool:
    for claim in result.claims:
        if claim.support_status in FAILING_SUPPORT_STATUSES:
            return False
        if claim.provenance_issue:
            return False

    for coverage in result.question_coverage:
        if coverage.status == "missing":
            return False

    if not result.final_table_present:
        return False
    return result.final_summary_present


def claim_level_failure_reasons(result: ClaimLevelJudgeResult) -> list[str]:
    reasons: list[str] = []
    for claim in result.claims:
        if claim.support_status in FAILING_SUPPORT_STATUSES:
            reasons.append(f"claim support_status={claim.support_status!r}: {claim.claim}")
        if claim.provenance_issue:
            reasons.append(f"provenance_issue: {claim.claim}")
    for coverage in result.question_coverage:
        if coverage.status == "missing":
            reasons.append(f"question {coverage.question_number} missing")
    if not result.final_table_present:
        reasons.append("final table missing")
    if not result.final_summary_present:
        reasons.append("final summary missing")
    return reasons


def _require_object(payload: dict[str, object], key: str) -> dict[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        msg = f"expected object field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for nested_key, item in cast(list[tuple[object, object]], list(value.items())):
        if not isinstance(nested_key, str):
            msg = f"expected string keys in {key!r}, got {type(nested_key).__name__}"
            raise TypeError(msg)
        result[nested_key] = item
    return result


def _require_non_empty_string(payload: dict[str, object], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    if value.strip() == "":
        msg = f"field {key!r} must be a non-empty string"
        raise ValueError(msg)
    return value


def _require_bool(payload: dict[str, object], key: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"expected bool field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _parse_string_list(payload: dict[str, object], key: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        msg = f"expected list field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    items: list[str] = []
    for index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, str):
            msg = f"{key}[{index}] must be a string, got {type(item).__name__}"
            raise TypeError(msg)
        if item.strip() == "":
            msg = f"{key}[{index}] must be a non-empty string"
            raise ValueError(msg)
        items.append(item)
    return items


def _parse_claim_item(item: object, index: int) -> ClaimAuditItem:
    if not isinstance(item, dict):
        msg = f"claims[{index}] must be an object, got {type(item).__name__}"
        raise TypeError(msg)
    payload: dict[str, object] = {}
    for key, value in cast(list[tuple[object, object]], list(item.items())):
        if not isinstance(key, str):
            msg = f"claims[{index}] expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        payload[key] = value

    support_status = _require_non_empty_string(payload, "support_status")
    if support_status not in ALLOWED_SUPPORT_STATUSES:
        msg = (
            f"claims[{index}].support_status must be one of "
            f"{sorted(ALLOWED_SUPPORT_STATUSES)}, got {support_status!r}"
        )
        raise ValueError(msg)

    fact_ids = _parse_string_list(payload, "reference_fact_ids")
    unknown = [fact_id for fact_id in fact_ids if fact_id not in KNOWN_REFERENCE_FACT_IDS]
    if unknown:
        msg = f"claims[{index}].reference_fact_ids contains unknown IDs: {unknown}"
        raise ValueError(msg)

    return ClaimAuditItem(
        claim=_require_non_empty_string(payload, "claim"),
        candidate_excerpt=_require_non_empty_string(payload, "candidate_excerpt"),
        support_status=support_status,
        reference_fact_ids=fact_ids,
        provenance_issue=_require_bool(payload, "provenance_issue"),
        explanation=_require_non_empty_string(payload, "explanation"),
    )


def _parse_coverage_item(item: object, index: int) -> QuestionCoverageItem:
    if not isinstance(item, dict):
        msg = f"question_coverage[{index}] must be an object, got {type(item).__name__}"
        raise TypeError(msg)
    payload: dict[str, object] = {}
    for key, value in cast(list[tuple[object, object]], list(item.items())):
        if not isinstance(key, str):
            msg = f"question_coverage[{index}] expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        payload[key] = value

    number_raw = payload.get("question_number")
    if isinstance(number_raw, bool) or not isinstance(number_raw, int):
        msg = (
            f"question_coverage[{index}].question_number must be an int, "
            f"got {type(number_raw).__name__}"
        )
        raise TypeError(msg)

    status = _require_non_empty_string(payload, "status")
    if status not in ALLOWED_COVERAGE_STATUSES:
        msg = (
            f"question_coverage[{index}].status must be one of "
            f"{sorted(ALLOWED_COVERAGE_STATUSES)}, got {status!r}"
        )
        raise ValueError(msg)

    return QuestionCoverageItem(
        question_number=number_raw,
        status=status,
        explanation=_require_non_empty_string(payload, "explanation"),
    )


def _validate_coverage(coverage: list[QuestionCoverageItem]) -> None:
    numbers = [item.question_number for item in coverage]
    expected = list(EXPECTED_QUESTION_NUMBERS)
    if sorted(numbers) != expected:
        missing = sorted(set(expected) - set(numbers))
        extras = sorted(set(numbers) - set(expected))
        duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
        msg = (
            "question_coverage must contain exactly question numbers 1..40 once each; "
            f"missing={missing}, extras={extras}, duplicates={duplicates}"
        )
        raise ValueError(msg)


def parse_claim_level_judge_result(text: str) -> ClaimLevelJudgeResult:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"judge output is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    if not isinstance(raw, dict):
        msg = f"expected JSON object, got {type(raw).__name__}"
        raise TypeError(msg)

    payload: dict[str, object] = {}
    for key, item in cast(list[tuple[object, object]], list(raw.items())):
        if not isinstance(key, str):
            msg = f"expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        payload[key] = item

    claims_raw = payload.get("claims")
    if not isinstance(claims_raw, list):
        msg = f"expected list field 'claims', got {type(claims_raw).__name__}"
        raise TypeError(msg)
    if not claims_raw:
        msg = "claims must be a non-empty list"
        raise ValueError(msg)

    claims: list[ClaimAuditItem] = []
    for index, item in enumerate(cast(list[object], claims_raw)):
        claims.append(_parse_claim_item(item, index))

    coverage_raw = payload.get("question_coverage")
    if not isinstance(coverage_raw, list):
        msg = f"expected list field 'question_coverage', got {type(coverage_raw).__name__}"
        raise TypeError(msg)

    coverage: list[QuestionCoverageItem] = []
    for index, item in enumerate(cast(list[object], coverage_raw)):
        coverage.append(_parse_coverage_item(item, index))
    _validate_coverage(coverage)

    return ClaimLevelJudgeResult(
        claims=claims,
        question_coverage=coverage,
        final_table_present=_require_bool(payload, "final_table_present"),
        final_summary_present=_require_bool(payload, "final_summary_present"),
        rationale=_require_non_empty_string(payload, "rationale"),
    )
