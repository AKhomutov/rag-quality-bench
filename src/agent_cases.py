from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


@dataclass(frozen=True)
class AgentToolSelectionCase:
    id: str
    category: str
    expected_behavior: str
    failure_description: str
    documents: list[str]
    question: str
    expected_answer_contains_any: list[str]
    forbidden_tools: list[str]
    preferred_tools: list[str]


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


def load_agent_tool_selection_cases(path: Path) -> list[AgentToolSelectionCase]:
    with path.open(encoding="utf-8") as file:
        raw = cast(object, yaml.safe_load(file))

    root = _validate_yaml_object(raw)
    cases_raw = _get_required_list(root, "cases")

    cases: list[AgentToolSelectionCase] = []

    for item in cases_raw:
        case_data = _validate_yaml_object(item)

        documents = _get_required_string_list(case_data, "documents")
        if not documents:
            msg = f"case {case_data.get('id')!r} must declare at least one document"
            raise ValueError(msg)

        case = AgentToolSelectionCase(
            id=_get_required_string(case_data, "id"),
            category=_get_required_string(case_data, "category"),
            expected_behavior=_get_required_string(case_data, "expected_behavior"),
            failure_description=_get_required_string(case_data, "failure_description"),
            documents=documents,
            question=_get_required_string(case_data, "question"),
            expected_answer_contains_any=_get_required_string_list(
                case_data,
                "expected_answer_contains_any",
            ),
            forbidden_tools=_get_required_string_list(case_data, "forbidden_tools"),
            preferred_tools=_get_required_string_list(case_data, "preferred_tools"),
        )
        cases.append(case)

    return cases
