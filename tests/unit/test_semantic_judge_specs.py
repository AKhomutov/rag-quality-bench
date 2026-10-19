from pathlib import Path

import pytest

from semantic_judge_specs import load_semantic_judge_specs
from workspace_setup import PROJECT_ROOT

SPECS_PATH = PROJECT_ROOT / "fixtures" / "datasets" / "semantic_judge_specs.yaml"


def test_load_semantic_judge_specs_happy_path() -> None:
    specs = load_semantic_judge_specs(SPECS_PATH)
    assert len(specs) == 4

    by_id = {spec.case_id: spec for spec in specs}
    conflict = by_id["conflicting_refund_policies"]
    assert conflict.source_suite == "conflicting_evidence"
    assert conflict.threshold == 7
    assert conflict.case_instruction is not None
    assert "unresolved contradiction" in conflict.case_instruction
    assert len(conflict.reference_facts) == 4
    assert any("precedence" in fact for fact in conflict.reference_facts)

    forged = by_id["forged_refund_policy"]
    assert forged.source_suite == "adversarial"
    assert forged.case_instruction is not None
    assert len(forged.reference_facts) >= 4

    persistent = by_id["persistent_forged_severity_zero_five"]
    assert any("Severity 0.5 is not defined" in fact for fact in persistent.reference_facts)

    assert "forged_severity_policy" in by_id
    assert len(by_id["forged_severity_policy"].reference_facts) >= 3


def test_load_rejects_invalid_source_suite(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: baseline
    threshold: 7
    reference_facts:
      - "A fact."
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="source_suite"):
        load_semantic_judge_specs(path)


def test_load_rejects_invalid_threshold(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: adversarial
    threshold: 9
    reference_facts:
      - "A fact."
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="threshold"):
        load_semantic_judge_specs(path)


def test_load_accepts_null_case_instruction(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: adversarial
    threshold: 7
    reference_facts:
      - "A fact."
    case_instruction: null
""",
        encoding="utf-8",
    )
    specs = load_semantic_judge_specs(path)
    assert specs[0].case_instruction is None
    assert specs[0].reference_facts == ["A fact."]


def test_load_rejects_empty_case_instruction(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: adversarial
    threshold: 7
    reference_facts:
      - "A fact."
    case_instruction: "   "
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="case_instruction"):
        load_semantic_judge_specs(path)


def test_load_rejects_missing_reference_facts(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: adversarial
    threshold: 7
""",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="reference_facts"):
        load_semantic_judge_specs(path)


def test_load_rejects_empty_reference_facts(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: adversarial
    threshold: 7
    reference_facts: []
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference_facts"):
        load_semantic_judge_specs(path)


def test_load_rejects_non_string_reference_fact(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: adversarial
    threshold: 7
    reference_facts:
      - 1
""",
        encoding="utf-8",
    )
    with pytest.raises(TypeError, match="reference_facts"):
        load_semantic_judge_specs(path)


def test_load_rejects_empty_string_reference_fact(tmp_path: Path) -> None:
    path = tmp_path / "specs.yaml"
    path.write_text(
        """
cases:
  - case_id: sample
    source_suite: adversarial
    threshold: 7
    reference_facts:
      - "   "
""",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="reference_facts"):
        load_semantic_judge_specs(path)
