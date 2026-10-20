"""Apply dependency-graph normalization offline to a saved split-claim summary.

No Ollama / live model calls.
"""

from __future__ import annotations

import json
from typing import cast

from split_claim_dependency_graph import (
    edges_from_dependency_map,
    normalize_dependency_graph,
)
from workspace_setup import PROJECT_ROOT

DEFAULT_SUMMARY = (
    PROJECT_ROOT / "reports" / "judge_debug" / "20261020T134259_split_claim" / "summary.json"
)


def _as_object(raw: object, context: str) -> dict[str, object]:
    if not isinstance(raw, dict):
        msg = f"{context}: expected object, got {type(raw).__name__}"
        raise TypeError(msg)
    result: dict[str, object] = {}
    for key, value in cast(list[tuple[object, object]], list(raw.items())):
        if not isinstance(key, str):
            msg = f"{context}: expected string keys"
            raise TypeError(msg)
        result[key] = value
    return result


def main() -> None:
    summary_path = DEFAULT_SUMMARY
    loaded: object = json.loads(summary_path.read_text(encoding="utf-8"))
    payload = _as_object(loaded, "summary")
    finals_raw = payload.get("final_claims")
    if not isinstance(finals_raw, list):
        msg = "summary.final_claims must be a list"
        raise TypeError(msg)

    before_derived = 0
    before_nie = 0
    depends_map: dict[str, list[str]] = {}
    status_by_id: dict[str, str] = {}

    for index, item in enumerate(cast(list[object], finals_raw)):
        claim = _as_object(item, f"final_claims[{index}]")
        claim_id_raw = claim.get("claim_id")
        status_raw = claim.get("status")
        depends_raw = claim.get("depends_on")
        if not isinstance(claim_id_raw, str) or not isinstance(status_raw, str):
            msg = f"final_claims[{index}] missing claim_id/status"
            raise TypeError(msg)
        if not isinstance(depends_raw, list):
            msg = f"final_claims[{index}].depends_on must be a list"
            raise TypeError(msg)
        depends: list[str] = []
        for dep in cast(list[object], depends_raw):
            if not isinstance(dep, str):
                msg = f"final_claims[{index}].depends_on must be strings"
                raise TypeError(msg)
            depends.append(dep)
        status_by_id[claim_id_raw] = status_raw
        if status_raw == "derived_from_unsupported_claim":
            before_derived += 1
            depends_map[claim_id_raw] = depends
        elif status_raw == "not_in_evidence":
            before_nie += 1
            if depends:
                depends_map[claim_id_raw] = depends

    raw_edges = edges_from_dependency_map(depends_map)
    graph = normalize_dependency_graph(raw_edges)

    after_derived = 0
    after_nie = before_nie
    downgraded: list[str] = []
    for claim_id, status in status_by_id.items():
        if status != "derived_from_unsupported_claim":
            continue
        normalized = graph.normalized_depends_on.get(claim_id, [])
        if normalized:
            after_derived += 1
        else:
            after_nie += 1
            downgraded.append(claim_id)

    print(f"Offline source: {summary_path.relative_to(PROJECT_ROOT)}")
    print("No live model call.")
    print()
    print("before normalization:")
    print(f"  derived count: {before_derived}")
    print(f"  not_in_evidence count: {before_nie}")
    print(f"  dependency edge count: {len(graph.raw_edges)}")
    print(f"  cycles: {len(graph.cycles)}")
    for cycle in graph.cycles:
        print(f"    Cycle: {' <-> '.join(cycle)}")
    print()
    print("after normalization:")
    print(f"  derived count: {after_derived}")
    print(f"  not_in_evidence count: {after_nie}")
    print(f"  dependency edge count: {len(graph.normalized_edges)}")
    print(f"  cycles=0 (normalized internal edges removed: {len(graph.removed_internal_edges)})")
    print(f"  downgraded derived -> not_in_evidence: {sorted(downgraded)}")
    print()

    for claim_id in ("C21", "C35"):
        if claim_id not in status_by_id:
            continue
        raw = depends_map.get(claim_id, [])
        normalized = graph.normalized_depends_on.get(claim_id, [])
        raw_status = status_by_id[claim_id]
        final_status = (
            "derived_from_unsupported_claim"
            if raw_status == "derived_from_unsupported_claim" and normalized
            else (
                "not_in_evidence" if raw_status == "derived_from_unsupported_claim" else raw_status
            )
        )
        print(f"{claim_id}:")
        print(f"  raw status: {raw_status}")
        print(f"  raw depends_on: {raw}")
        print(f"  normalized depends_on: {normalized}")
        print(f"  final status: {final_status}")
        print(f"  in cycle: {claim_id in graph.claims_in_cycles}")


if __name__ == "__main__":
    main()
