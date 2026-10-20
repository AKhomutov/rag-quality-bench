from __future__ import annotations

import pytest

from split_claim_dependency_graph import (
    DependencyEdge,
    detect_cycles,
    edges_from_dependency_map,
    normalize_dependency_graph,
    validate_raw_dependency_edges,
)
from split_claim_results import (
    BaseClaimJudgement,
    DependencyJudgement,
    ExtractedClaim,
    build_final_judgements,
)


def _edge(source: str, target: str) -> DependencyEdge:
    return DependencyEdge(source=source, target=target)


def test_no_cycles_unchanged() -> None:
    edges = [_edge("C3", "C1"), _edge("C5", "C3")]
    result = normalize_dependency_graph(edges)
    assert result.cycles == []
    assert {(e.source, e.target) for e in result.normalized_edges} == {
        ("C3", "C1"),
        ("C5", "C3"),
    }


def test_self_loop_removed() -> None:
    edges = [_edge("C3", "C3")]
    result = normalize_dependency_graph(edges)
    assert result.cycles == [["C3"]]
    assert result.self_loop_count == 1
    assert result.normalized_edges == []
    assert result.normalized_depends_on["C3"] == []


def test_two_node_cycle_removed() -> None:
    edges = [_edge("C1", "C2"), _edge("C2", "C1")]
    result = normalize_dependency_graph(edges)
    assert result.cycles == [["C1", "C2"]]
    assert result.normalized_edges == []
    assert result.normalized_depends_on["C1"] == []
    assert result.normalized_depends_on["C2"] == []


def test_cycle_with_external_anchor() -> None:
    edges = [_edge("C1", "C2"), _edge("C2", "C1"), _edge("C2", "C9")]
    result = normalize_dependency_graph(edges)
    assert result.cycles == [["C1", "C2"]]
    assert {(e.source, e.target) for e in result.normalized_edges} == {("C2", "C9")}
    assert result.normalized_depends_on["C1"] == []
    assert result.normalized_depends_on["C2"] == ["C9"]


def test_three_node_cycle_removed() -> None:
    edges = [_edge("C1", "C2"), _edge("C2", "C3"), _edge("C3", "C1")]
    result = normalize_dependency_graph(edges)
    assert result.cycles == [["C1", "C2", "C3"]]
    assert result.normalized_edges == []


def test_cycle_plus_multiple_external_edges() -> None:
    edges = [
        _edge("C1", "C2"),
        _edge("C2", "C1"),
        _edge("C1", "C9"),
        _edge("C2", "C8"),
    ]
    result = normalize_dependency_graph(edges)
    assert {(e.source, e.target) for e in result.normalized_edges} == {
        ("C1", "C9"),
        ("C2", "C8"),
    }


def test_detect_cycles_sorted() -> None:
    adjacency = {
        "C2": ["C1"],
        "C1": ["C2"],
        "C4": ["C4"],
    }
    cycles = detect_cycles(adjacency)
    assert cycles == [["C4"], ["C1", "C2"]]


def test_duplicate_raw_dependency_rejected() -> None:
    edges = [_edge("C1", "C2"), _edge("C1", "C2")]
    with pytest.raises(ValueError, match="duplicate"):
        validate_raw_dependency_edges(
            edges,
            known_claim_ids=frozenset({"C1", "C2"}),
        )


def test_unknown_claim_id_rejected() -> None:
    edges = [_edge("C1", "C99")]
    with pytest.raises(ValueError, match="unknown"):
        validate_raw_dependency_edges(
            edges,
            known_claim_ids=frozenset({"C1", "C2"}),
        )


def test_supported_premise_rejected() -> None:
    edges = [_edge("C3", "C2")]
    with pytest.raises(ValueError, match="premise pool"):
        validate_raw_dependency_edges(
            edges,
            known_claim_ids=frozenset({"C2", "C3"}),
            allowed_premise_ids=frozenset({"C3"}),
        )


def test_build_final_downgrades_pure_cycle() -> None:
    extracted = [
        ExtractedClaim("C1", "a", "a"),
        ExtractedClaim("C2", "b", "b"),
        ExtractedClaim("C9", "anchor", "anchor"),
    ]
    base = {
        "C1": BaseClaimJudgement("C1", "not_in_evidence", [], False, "u1"),
        "C2": BaseClaimJudgement("C2", "not_in_evidence", [], False, "u2"),
        "C9": BaseClaimJudgement("C9", "contradicted", ["F1"], True, "fake"),
    }
    deps = {
        "C1": DependencyJudgement("C1", True, ["C2"], "cycle"),
        "C2": DependencyJudgement("C2", True, ["C1", "C9"], "cycle+anchor"),
    }
    bundle = build_final_judgements(extracted, base, deps)
    by_id = {item.claim_id: item for item in bundle.finals}
    assert by_id["C1"].status == "not_in_evidence"
    assert by_id["C1"].raw_depends_on == ["C2"]
    assert by_id["C1"].depends_on == []
    assert by_id["C1"].dependency_cycle is True
    assert by_id["C2"].status == "derived_from_unsupported_claim"
    assert by_id["C2"].depends_on == ["C9"]
    assert by_id["C2"].raw_depends_on == ["C1", "C9"]
    assert "C1" in bundle.downgraded_claims
    assert "C2" not in bundle.downgraded_claims


def test_edges_from_dependency_map_order() -> None:
    edges = edges_from_dependency_map({"C2": ["C1"], "C1": ["C3"]})
    assert [(e.source, e.target) for e in edges] == [("C1", "C3"), ("C2", "C1")]
