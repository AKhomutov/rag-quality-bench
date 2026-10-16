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


@dataclass(frozen=True)
class RagCase:
    id: str
    category: str
    question: str
    expected_answer_contains_any: list[str]
    expected_source_contains: str | None
    expected_answer_excludes: str | None


def _get_optional_string(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)

    if value is None:
        return None

    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return value


def load_rag_cases(path: Path) -> list[RagCase]:
    with path.open(encoding="utf-8") as file:
        raw = cast(object, yaml.safe_load(file))

    root = validate_yaml_object(raw)
    cases_raw = get_required_list(root, "cases")

    cases: list[RagCase] = []

    for item in cases_raw:
        case_data = validate_yaml_object(item)

        case = RagCase(
            id=get_required_string(case_data, "id"),
            category=get_required_string(case_data, "category"),
            question=get_required_string(case_data, "question"),
            expected_answer_contains_any=get_required_string_list(
                case_data,
                "expected_answer_contains_any",
            ),
            expected_source_contains=_get_optional_string(
                case_data,
                "expected_source_contains",
            ),
            expected_answer_excludes=_get_optional_string(
                case_data,
                "expected_answer_excludes",
            ),
        )

        cases.append(case)

    return cases
