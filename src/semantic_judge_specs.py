from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from yaml_validation import (
    get_required_list,
    get_required_string,
    get_required_string_list,
    validate_yaml_object,
)

ALLOWED_SOURCE_SUITES = frozenset({"adversarial", "conflicting_evidence"})


@dataclass(frozen=True)
class SemanticJudgeSpec:
    case_id: str
    source_suite: str
    threshold: int
    reference_facts: list[str]
    case_instruction: str | None


def _get_required_threshold(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"expected int field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)
    if value < 0 or value > 8:
        msg = f"threshold must be in 0..8, got {value}"
        raise ValueError(msg)
    return value


def _get_required_reference_facts(mapping: dict[str, object]) -> list[str]:
    facts = get_required_string_list(mapping, "reference_facts")
    if not facts:
        msg = "reference_facts must be a non-empty list"
        raise ValueError(msg)
    for index, fact in enumerate(facts):
        if fact.strip() == "":
            msg = f"reference_facts[{index}] must be a non-empty string"
            raise ValueError(msg)
    return facts


def _get_optional_case_instruction(mapping: dict[str, object]) -> str | None:
    if "case_instruction" not in mapping:
        return None
    value = mapping["case_instruction"]
    if value is None:
        return None
    if not isinstance(value, str):
        msg = f"expected string field 'case_instruction', got {type(value).__name__}"
        raise TypeError(msg)
    if value.strip() == "":
        msg = "case_instruction must be a non-empty string when provided"
        raise ValueError(msg)
    return value


def load_semantic_judge_specs(path: Path) -> list[SemanticJudgeSpec]:
    with path.open(encoding="utf-8") as file:
        raw = cast(object, yaml.safe_load(file))

    root = validate_yaml_object(raw)
    cases_raw = get_required_list(root, "cases")

    specs: list[SemanticJudgeSpec] = []
    for item in cases_raw:
        case_data = validate_yaml_object(item)
        source_suite = get_required_string(case_data, "source_suite")
        if source_suite not in ALLOWED_SOURCE_SUITES:
            msg = (
                f"source_suite must be one of {sorted(ALLOWED_SOURCE_SUITES)}, got {source_suite!r}"
            )
            raise ValueError(msg)

        specs.append(
            SemanticJudgeSpec(
                case_id=get_required_string(case_data, "case_id"),
                source_suite=source_suite,
                threshold=_get_required_threshold(case_data, "threshold"),
                reference_facts=_get_required_reference_facts(case_data),
                case_instruction=_get_optional_case_instruction(case_data),
            )
        )

    return specs
