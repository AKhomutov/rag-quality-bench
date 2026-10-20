"""DTOs, parsers, and Python aggregation for the split claim pipeline."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import cast

from long_form_severity_facts import KNOWN_REFERENCE_FACT_IDS
from split_claim_dependency_graph import (
    DependencyGraphNormalization,
    edges_from_dependency_map,
    normalize_dependency_graph,
    validate_raw_dependency_edges,
)

CLAIM_ID_PATTERN = re.compile(r"^C\d+$")

BASE_CLAIM_STATUSES = frozenset(
    {
        "supported",
        "contradicted",
        "not_in_evidence",
    }
)

FINAL_CLAIM_STATUSES = frozenset(
    {
        "supported",
        "contradicted",
        "not_in_evidence",
        "derived_from_unsupported_claim",
    }
)

FAILING_FINAL_STATUSES = frozenset(
    {
        "contradicted",
        "not_in_evidence",
        "derived_from_unsupported_claim",
    }
)


@dataclass(frozen=True)
class ExtractedClaim:
    claim_id: str
    claim: str
    candidate_excerpt: str


@dataclass(frozen=True)
class BaseClaimJudgement:
    claim_id: str
    status: str
    reference_fact_ids: list[str]
    provenance_issue: bool
    explanation: str


@dataclass(frozen=True)
class DependencyJudgement:
    claim_id: str
    derived: bool
    depends_on: list[str]
    explanation: str


@dataclass(frozen=True)
class FinalClaimJudgement:
    claim_id: str
    claim: str
    candidate_excerpt: str
    status: str
    reference_fact_ids: list[str]
    provenance_issue: bool
    depends_on: list[str]
    raw_depends_on: list[str]
    dependency_cycle: bool
    explanation: str


@dataclass(frozen=True)
class FinalJudgementBundle:
    finals: list[FinalClaimJudgement]
    graph: DependencyGraphNormalization
    downgraded_claims: list[str]


@dataclass(frozen=True)
class SplitClaimMetrics:
    total_claims: int
    supported: int
    contradicted: int
    not_in_evidence: int
    derived_from_unsupported_claim: int
    provenance_issues: int
    grounded_claim_rate: float


def _as_object(raw: object, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        msg = f"{context}: expected JSON object, got {type(raw).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, item in cast(list[tuple[object, object]], list(raw.items())):
        if not isinstance(key, str):
            msg = f"{context}: expected string keys, got {type(key).__name__}"
            raise TypeError(msg)
        result[key] = item
    return result


def _require_non_empty_string(payload: dict[str, object], key: str, context: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str):
        msg = f"{context}: expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    if value.strip() == "":
        msg = f"{context}: field {key!r} must be a non-empty string"
        raise ValueError(msg)
    return value


def _require_bool(payload: dict[str, object], key: str, context: str) -> bool:
    value = payload.get(key)
    if not isinstance(value, bool):
        msg = f"{context}: expected bool field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    return value


def _parse_string_list(payload: dict[str, object], key: str, context: str) -> list[str]:
    value = payload.get(key)
    if not isinstance(value, list):
        msg = f"{context}: expected list field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    items: list[str] = []
    for index, item in enumerate(cast(list[object], value)):
        if not isinstance(item, str):
            msg = f"{context}: {key}[{index}] must be a string, got {type(item).__name__}"
            raise TypeError(msg)
        if item.strip() == "":
            msg = f"{context}: {key}[{index}] must be a non-empty string"
            raise ValueError(msg)
        items.append(item)
    return items


def _validate_claim_id(claim_id: str, context: str) -> None:
    if CLAIM_ID_PATTERN.fullmatch(claim_id) is None:
        msg = f"{context}: claim_id must match C<number>, got {claim_id!r}"
        raise ValueError(msg)


def canonicalize_extracted_claim_ids(claims: list[ExtractedClaim]) -> list[ExtractedClaim]:
    """Rewrite claim IDs to contiguous C1..Cn without changing claim text."""
    return [
        ExtractedClaim(
            claim_id=f"C{index}",
            claim=item.claim,
            candidate_excerpt=item.candidate_excerpt,
        )
        for index, item in enumerate(claims, start=1)
    ]


def parse_extracted_claims(text: str) -> list[ExtractedClaim]:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"extraction output is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    payload = _as_object(raw, "extraction")
    claims_raw = payload.get("claims")
    if not isinstance(claims_raw, list):
        msg = f"extraction: expected list field 'claims', got {type(claims_raw).__name__}"
        raise TypeError(msg)
    if not claims_raw:
        msg = "extraction: claims must be a non-empty list"
        raise ValueError(msg)

    claims: list[ExtractedClaim] = []
    seen_ids: set[str] = set()
    for index, item in enumerate(cast(list[object], claims_raw)):
        context = f"extraction.claims[{index}]"
        item_payload = _as_object(item, context)
        claim_id = _require_non_empty_string(item_payload, "claim_id", context)
        _validate_claim_id(claim_id, context)
        if claim_id in seen_ids:
            msg = f"{context}: duplicate claim_id {claim_id!r}"
            raise ValueError(msg)
        seen_ids.add(claim_id)
        claims.append(
            ExtractedClaim(
                claim_id=claim_id,
                claim=_require_non_empty_string(item_payload, "claim", context),
                candidate_excerpt=_require_non_empty_string(
                    item_payload, "candidate_excerpt", context
                ),
            )
        )
    return claims


def partition_into_batches[T](items: list[T], batch_size: int) -> list[list[T]]:
    if batch_size < 1:
        msg = f"batch_size must be >= 1, got {batch_size}"
        raise ValueError(msg)
    if not items:
        return []
    return [items[index : index + batch_size] for index in range(0, len(items), batch_size)]


def _parse_base_claim_payload(
    payload: dict[str, object],
    *,
    context: str,
    expected_claim_id: str | None = None,
) -> BaseClaimJudgement:
    claim_id = _require_non_empty_string(payload, "claim_id", context)
    if expected_claim_id is not None and claim_id != expected_claim_id:
        msg = f"{context}: claim_id mismatch: expected {expected_claim_id!r}, got {claim_id!r}"
        raise ValueError(msg)

    status = _require_non_empty_string(payload, "status", context)
    if status not in BASE_CLAIM_STATUSES:
        msg = f"{context}: status must be one of {sorted(BASE_CLAIM_STATUSES)}, got {status!r}"
        raise ValueError(msg)

    fact_ids = _parse_string_list(payload, "reference_fact_ids", context)
    unknown = [fact_id for fact_id in fact_ids if fact_id not in KNOWN_REFERENCE_FACT_IDS]
    if unknown:
        msg = f"{context}: reference_fact_ids contains unknown IDs: {unknown}"
        raise ValueError(msg)

    return BaseClaimJudgement(
        claim_id=claim_id,
        status=status,
        reference_fact_ids=fact_ids,
        provenance_issue=_require_bool(payload, "provenance_issue", context),
        explanation=_require_non_empty_string(payload, "explanation", context),
    )


def _parse_dependency_payload(
    payload: dict[str, object],
    *,
    context: str,
    expected_claim_id: str | None,
    known_claim_ids: frozenset[str],
) -> DependencyJudgement:
    claim_id = _require_non_empty_string(payload, "claim_id", context)
    if expected_claim_id is not None and claim_id != expected_claim_id:
        msg = f"{context}: claim_id mismatch: expected {expected_claim_id!r}, got {claim_id!r}"
        raise ValueError(msg)

    derived = _require_bool(payload, "derived", context)
    depends_on = _parse_string_list(payload, "depends_on", context)

    if not derived and depends_on:
        msg = f"{context}: depends_on must be empty when derived=false"
        raise ValueError(msg)
    if derived and not depends_on:
        msg = f"{context}: depends_on must be non-empty when derived=true"
        raise ValueError(msg)

    if len(depends_on) != len(set(depends_on)):
        duplicates = sorted({dep_id for dep_id in depends_on if depends_on.count(dep_id) > 1})
        msg = f"{context}: depends_on contains duplicate IDs: {duplicates}"
        raise ValueError(msg)

    for dep_id in depends_on:
        _validate_claim_id(dep_id, context)
        if dep_id not in known_claim_ids:
            msg = f"{context}: depends_on contains unknown claim_id {dep_id!r}"
            raise ValueError(msg)
        # Self-loops are allowed; graph normalization handles them.

    return DependencyJudgement(
        claim_id=claim_id,
        derived=derived,
        depends_on=depends_on,
        explanation=_require_non_empty_string(payload, "explanation", context),
    )


def _require_exact_claim_coverage(
    *,
    expected_ids: frozenset[str],
    observed_ids: list[str],
    context: str,
) -> None:
    observed_set = set(observed_ids)
    if len(observed_ids) != len(observed_set):
        duplicates = sorted(
            {claim_id for claim_id in observed_ids if observed_ids.count(claim_id) > 1}
        )
        msg = f"{context}: duplicate claim IDs: {duplicates}"
        raise ValueError(msg)
    missing = sorted(expected_ids - observed_set)
    unknown = sorted(observed_set - expected_ids)
    if missing or unknown:
        msg = f"{context}: claim ID coverage mismatch; missing={missing}, unknown={unknown}"
        raise ValueError(msg)


def parse_base_claim_judgement(text: str, *, expected_claim_id: str) -> BaseClaimJudgement:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"base claim judge output is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    context = f"base claim {expected_claim_id}"
    payload = _as_object(raw, context)
    return _parse_base_claim_payload(
        payload,
        context=context,
        expected_claim_id=expected_claim_id,
    )


def parse_base_claim_batch(
    text: str,
    *,
    expected_claim_ids: frozenset[str],
) -> list[BaseClaimJudgement]:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"base claim batch output is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    context = "base claim batch"
    payload = _as_object(raw, context)
    judgements_raw = payload.get("judgements")
    if not isinstance(judgements_raw, list):
        msg = f"{context}: expected list field 'judgements', got {type(judgements_raw).__name__}"
        raise TypeError(msg)

    judgements: list[BaseClaimJudgement] = []
    observed_ids: list[str] = []
    for index, item in enumerate(cast(list[object], judgements_raw)):
        item_context = f"{context}.judgements[{index}]"
        item_payload = _as_object(item, item_context)
        judgement = _parse_base_claim_payload(item_payload, context=item_context)
        judgements.append(judgement)
        observed_ids.append(judgement.claim_id)

    _require_exact_claim_coverage(
        expected_ids=expected_claim_ids,
        observed_ids=observed_ids,
        context=context,
    )
    return judgements


def parse_dependency_judgement(
    text: str,
    *,
    expected_claim_id: str,
    known_claim_ids: frozenset[str],
) -> DependencyJudgement:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"dependency judge output is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    context = f"dependency {expected_claim_id}"
    payload = _as_object(raw, context)
    return _parse_dependency_payload(
        payload,
        context=context,
        expected_claim_id=expected_claim_id,
        known_claim_ids=known_claim_ids,
    )


def parse_dependency_batch(
    text: str,
    *,
    expected_claim_ids: frozenset[str],
    known_claim_ids: frozenset[str],
) -> list[DependencyJudgement]:
    try:
        raw: object = json.loads(text)
    except json.JSONDecodeError as exc:
        msg = f"dependency batch output is not valid JSON: {exc}"
        raise ValueError(msg) from exc

    context = "dependency batch"
    payload = _as_object(raw, context)
    dependencies_raw = payload.get("dependencies")
    if not isinstance(dependencies_raw, list):
        msg = (
            f"{context}: expected list field 'dependencies', got {type(dependencies_raw).__name__}"
        )
        raise TypeError(msg)

    judgements: list[DependencyJudgement] = []
    observed_ids: list[str] = []
    for index, item in enumerate(cast(list[object], dependencies_raw)):
        item_context = f"{context}.dependencies[{index}]"
        item_payload = _as_object(item, item_context)
        judgement = _parse_dependency_payload(
            item_payload,
            context=item_context,
            expected_claim_id=None,
            known_claim_ids=known_claim_ids,
        )
        judgements.append(judgement)
        observed_ids.append(judgement.claim_id)

    _require_exact_claim_coverage(
        expected_ids=expected_claim_ids,
        observed_ids=observed_ids,
        context=context,
    )
    return judgements


def build_final_judgements(
    extracted: list[ExtractedClaim],
    base_by_id: dict[str, BaseClaimJudgement],
    dependency_by_id: dict[str, DependencyJudgement],
) -> FinalJudgementBundle:
    known_ids = frozenset(item.claim_id for item in extracted)
    premise_ids = frozenset(
        judgement.claim_id
        for judgement in base_by_id.values()
        if judgement.status in {"contradicted", "not_in_evidence"}
    )

    raw_depends_map: dict[str, list[str]] = {}
    for claim_id, dependency in dependency_by_id.items():
        raw_depends_map[claim_id] = list(dependency.depends_on)

    raw_edges = edges_from_dependency_map(raw_depends_map)
    validate_raw_dependency_edges(
        raw_edges,
        known_claim_ids=known_ids,
        allowed_premise_ids=premise_ids,
    )
    graph = normalize_dependency_graph(raw_edges)

    finals: list[FinalClaimJudgement] = []
    downgraded: list[str] = []
    for item in extracted:
        base = base_by_id[item.claim_id]
        in_cycle = graph.cycle_membership.get(item.claim_id, False)
        if base.status in {"supported", "contradicted"}:
            finals.append(
                FinalClaimJudgement(
                    claim_id=item.claim_id,
                    claim=item.claim,
                    candidate_excerpt=item.candidate_excerpt,
                    status=base.status,
                    reference_fact_ids=list(base.reference_fact_ids),
                    provenance_issue=base.provenance_issue,
                    depends_on=[],
                    raw_depends_on=[],
                    dependency_cycle=False,
                    explanation=base.explanation,
                )
            )
            continue

        # base.status == not_in_evidence
        dep = dependency_by_id.get(item.claim_id)
        raw_depends = list(dep.depends_on) if dep is not None else []
        normalized_depends = list(graph.normalized_depends_on.get(item.claim_id, []))

        if dep is not None and dep.derived:
            if normalized_depends:
                finals.append(
                    FinalClaimJudgement(
                        claim_id=item.claim_id,
                        claim=item.claim,
                        candidate_excerpt=item.candidate_excerpt,
                        status="derived_from_unsupported_claim",
                        reference_fact_ids=list(base.reference_fact_ids),
                        provenance_issue=base.provenance_issue,
                        depends_on=normalized_depends,
                        raw_depends_on=raw_depends,
                        dependency_cycle=in_cycle,
                        explanation=dep.explanation,
                    )
                )
            else:
                downgraded.append(item.claim_id)
                finals.append(
                    FinalClaimJudgement(
                        claim_id=item.claim_id,
                        claim=item.claim,
                        candidate_excerpt=item.candidate_excerpt,
                        status="not_in_evidence",
                        reference_fact_ids=list(base.reference_fact_ids),
                        provenance_issue=base.provenance_issue,
                        depends_on=[],
                        raw_depends_on=raw_depends,
                        dependency_cycle=in_cycle,
                        explanation=dep.explanation,
                    )
                )
        else:
            explanation = base.explanation
            if dep is not None:
                explanation = dep.explanation
            finals.append(
                FinalClaimJudgement(
                    claim_id=item.claim_id,
                    claim=item.claim,
                    candidate_excerpt=item.candidate_excerpt,
                    status="not_in_evidence",
                    reference_fact_ids=list(base.reference_fact_ids),
                    provenance_issue=base.provenance_issue,
                    depends_on=normalized_depends,
                    raw_depends_on=raw_depends,
                    dependency_cycle=in_cycle,
                    explanation=explanation,
                )
            )
    return FinalJudgementBundle(
        finals=finals,
        graph=graph,
        downgraded_claims=downgraded,
    )


def split_claim_passed(finals: list[FinalClaimJudgement]) -> bool:
    for item in finals:
        if item.status in FAILING_FINAL_STATUSES:
            return False
        if item.provenance_issue:
            return False
    return True


def split_claim_failure_reasons(finals: list[FinalClaimJudgement]) -> list[str]:
    reasons: list[str] = []
    for item in finals:
        if item.status in FAILING_FINAL_STATUSES:
            reasons.append(f"claim {item.claim_id} status={item.status!r}: {item.claim}")
        if item.provenance_issue:
            reasons.append(f"claim {item.claim_id} provenance_issue: {item.claim}")
    return reasons


def compute_split_claim_metrics(finals: list[FinalClaimJudgement]) -> SplitClaimMetrics:
    total = len(finals)
    supported = sum(1 for item in finals if item.status == "supported")
    contradicted = sum(1 for item in finals if item.status == "contradicted")
    not_in_evidence = sum(1 for item in finals if item.status == "not_in_evidence")
    derived = sum(1 for item in finals if item.status == "derived_from_unsupported_claim")
    provenance = sum(1 for item in finals if item.provenance_issue)
    rate = 0.0 if total == 0 else supported / total
    return SplitClaimMetrics(
        total_claims=total,
        supported=supported,
        contradicted=contradicted,
        not_in_evidence=not_in_evidence,
        derived_from_unsupported_claim=derived,
        provenance_issues=provenance,
        grounded_claim_rate=rate,
    )


def unsupported_premise_pool(
    base_judgements: list[BaseClaimJudgement],
) -> list[BaseClaimJudgement]:
    """Claims usable as dependency premises: contradicted or not_in_evidence."""
    return [item for item in base_judgements if item.status in {"contradicted", "not_in_evidence"}]
