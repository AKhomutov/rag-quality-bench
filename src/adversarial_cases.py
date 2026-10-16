from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml


@dataclass(frozen=True)
class ConversationStep:
    message: str


@dataclass(frozen=True)
class AdversarialCase:
    id: str
    category: str
    expected_behavior: str
    failure_description: str
    steps: list[ConversationStep]
    expected_final_contains_all: list[str]
    expected_final_contains_any: list[str]
    expected_final_excludes: list[str]


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


def load_adversarial_cases(path: Path) -> list[AdversarialCase]:
    with path.open(encoding="utf-8") as file:
        raw = cast(object, yaml.safe_load(file))

    root = _validate_yaml_object(raw)
    cases_raw = _get_required_list(root, "cases")

    cases: list[AdversarialCase] = []

    for item in cases_raw:
        case_data = _validate_yaml_object(item)
        steps_raw = _get_required_list(case_data, "steps")

        steps: list[ConversationStep] = []
        for step_item in steps_raw:
            step_data = _validate_yaml_object(step_item)
            steps.append(
                ConversationStep(
                    message=_get_required_string(step_data, "message"),
                )
            )

        if not steps:
            msg = f"case {case_data.get('id')!r} must have at least one step"
            raise ValueError(msg)

        case = AdversarialCase(
            id=_get_required_string(case_data, "id"),
            category=_get_required_string(case_data, "category"),
            expected_behavior=_get_required_string(case_data, "expected_behavior"),
            failure_description=_get_required_string(case_data, "failure_description"),
            steps=steps,
            expected_final_contains_all=_get_required_string_list(
                case_data,
                "expected_final_contains_all",
            ),
            expected_final_contains_any=_get_required_string_list(
                case_data,
                "expected_final_contains_any",
            ),
            expected_final_excludes=_get_required_string_list(
                case_data,
                "expected_final_excludes",
            ),
        )
        cases.append(case)

    return cases
