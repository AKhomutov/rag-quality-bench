from pathlib import Path

from conflicting_cases import load_conflicting_evidence_cases

CASES_PATH = Path("fixtures/datasets/conflicting_evidence.yaml")


def test_conflicting_cases_load_without_lexical_expectations() -> None:
    cases = load_conflicting_evidence_cases(CASES_PATH)
    assert len(cases) == 1

    case = cases[0]
    assert case.id == "conflicting_refund_policies"
    assert case.documents == ["acme_support_policy.md", "acme_no_refund_policy.md"]
    assert len(case.steps) == 4
    assert not hasattr(case, "expected_final_contains_all")
    assert not hasattr(case, "expected_final_contains_any")
    assert not hasattr(case, "expected_final_excludes")
