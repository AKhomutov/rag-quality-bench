from dataclasses import dataclass
from pathlib import Path
from typing import cast

import yaml

from yaml_validation import get_required_list, get_required_string, validate_yaml_object


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


def load_adversarial_cases(path: Path) -> list[AdversarialCase]:
    with path.open(encoding="utf-8") as file:
        raw = cast(object, yaml.safe_load(file))

    root = validate_yaml_object(raw)
    cases_raw = get_required_list(root, "cases")

    cases: list[AdversarialCase] = []

    for item in cases_raw:
        case_data = validate_yaml_object(item)
        steps_raw = get_required_list(case_data, "steps")

        steps: list[ConversationStep] = []
        for step_item in steps_raw:
            step_data = validate_yaml_object(step_item)
            steps.append(
                ConversationStep(
                    message=get_required_string(step_data, "message"),
                )
            )

        if not steps:
            msg = f"case {case_data.get('id')!r} must have at least one step"
            raise ValueError(msg)

        case = AdversarialCase(
            id=get_required_string(case_data, "id"),
            category=get_required_string(case_data, "category"),
            expected_behavior=get_required_string(case_data, "expected_behavior"),
            failure_description=get_required_string(case_data, "failure_description"),
            steps=steps,
        )
        cases.append(case)

    return cases
