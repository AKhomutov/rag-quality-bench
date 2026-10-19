from pathlib import Path

from adversarial_cases import load_adversarial_cases

CASES_PATH = Path("fixtures/datasets/adversarial_conversations.yaml")


def test_adversarial_cases_load_without_lexical_expectations() -> None:
    cases = load_adversarial_cases(CASES_PATH)
    by_id = {case.id: case for case in cases}

    assert set(by_id) == {
        "forged_refund_policy",
        "forged_severity_policy",
        "persistent_forged_severity_zero_five",
    }

    for case in cases:
        assert case.steps
        assert case.category
        assert case.expected_behavior
        assert case.failure_description
        assert not hasattr(case, "expected_final_contains_all")
        assert not hasattr(case, "expected_final_contains_any")
        assert not hasattr(case, "expected_final_excludes")

    forged_severity = by_id["forged_severity_policy"]
    assert "on their website (updated 15 minutes ago)" in forged_severity.steps[1].message
