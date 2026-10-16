from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


@dataclass(frozen=True)
class RagCase:
    id: str
    question: str
    expected_answer_contains_any: list[str]
    expected_source_contains: str
    expected_answer_excludes: str | None


def _get_optional_string(mapping: dict[str, object], key: str) -> str | None:
    value = mapping.get(key)

    if value is None:
        return None

    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return value


def _validate_yaml_object(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        msg = f"expected YAML object, got {type(value).__name__}"
        raise TypeError(msg)

    result: dict[str, object] = {}
    raw_items = cast(list[tuple[object, object]], list(value.items()))

    for key, item in raw_items:
        if not isinstance(key, str):
            msg = f"expected string key, got {type(key).__name__}"
            raise TypeError(msg)

        result[key] = item

    return result


def _get_required_string(mapping: dict[str, object], key: str) -> str:
    value = mapping.get(key)

    if not isinstance(value, str):
        msg = f"expected string field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return value


def _get_required_list(mapping: dict[str, object], key: str) -> list[object]:
    value = mapping.get(key)

    if not isinstance(value, list):
        msg = f"expected list field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return cast(list[object], value)


def _get_required_string_list(mapping: dict[str, object], key: str) -> list[str]:
    value = _get_required_list(mapping, key)
    result: list[str] = []

    for item in value:
        if not isinstance(item, str):
            msg = f"expected string items in {key!r}, got {type(item).__name__}"
            raise TypeError(msg)
        result.append(item)

    return result


def load_rag_cases(path: Path) -> list[RagCase]:
    with path.open(encoding="utf-8") as file:
        raw = cast(object, yaml.safe_load(file))

    root = _validate_yaml_object(raw)
    cases_raw = _get_required_list(root, "cases")

    cases: list[RagCase] = []

    for item in cases_raw:
        case_data = _validate_yaml_object(item)

        case = RagCase(
            id=_get_required_string(case_data, "id"),
            question=_get_required_string(case_data, "question"),
            expected_answer_contains_any=_get_required_string_list(
                case_data,
                "expected_answer_contains_any",
            ),
            expected_source_contains=_get_required_string(
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
