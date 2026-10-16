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
    trials: int
    minimum_pass_rate: float


def _get_required_int(mapping: dict[str, object], key: str) -> int:
    value = mapping.get(key)

    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"expected int field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return value


def _get_required_float(mapping: dict[str, object], key: str) -> float:
    value = mapping.get(key)

    if not isinstance(value, float):
        msg = f"expected float field {key!r}, got {type(value).__name__}"
        raise TypeError(msg)

    return value


def load_agent_tool_selection_cases(path: Path) -> list[AgentToolSelectionCase]:
    with path.open(encoding="utf-8") as file:
        raw = cast(object, yaml.safe_load(file))

    root = validate_yaml_object(raw)
    cases_raw = get_required_list(root, "cases")

    cases: list[AgentToolSelectionCase] = []

    for item in cases_raw:
        case_data = validate_yaml_object(item)

        documents = get_required_string_list(case_data, "documents")
        if not documents:
            msg = f"case {case_data.get('id')!r} must declare at least one document"
            raise ValueError(msg)

        trials = _get_required_int(case_data, "trials")
        if trials < 1:
            msg = f"case {case_data.get('id')!r}: trials must be >= 1, got {trials}"
            raise ValueError(msg)

        minimum_pass_rate = _get_required_float(case_data, "minimum_pass_rate")
        if minimum_pass_rate < 0.0 or minimum_pass_rate > 1.0:
            msg = (
                f"case {case_data.get('id')!r}: minimum_pass_rate must be between "
                f"0.0 and 1.0, got {minimum_pass_rate}"
            )
            raise ValueError(msg)

        case = AgentToolSelectionCase(
            id=get_required_string(case_data, "id"),
            category=get_required_string(case_data, "category"),
            expected_behavior=get_required_string(case_data, "expected_behavior"),
            failure_description=get_required_string(case_data, "failure_description"),
            documents=documents,
            question=get_required_string(case_data, "question"),
            expected_answer_contains_any=get_required_string_list(
                case_data,
                "expected_answer_contains_any",
            ),
            forbidden_tools=get_required_string_list(case_data, "forbidden_tools"),
            preferred_tools=get_required_string_list(case_data, "preferred_tools"),
            trials=trials,
            minimum_pass_rate=minimum_pass_rate,
        )
        cases.append(case)

    return cases
