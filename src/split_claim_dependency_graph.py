"""Deterministic dependency-graph cycle detection and normalization.

Raw Stage 2B LLM edges are preserved. Normalized edges drop cycle-internal
dependencies so downstream statuses stay graph-consistent without another LLM call.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DependencyEdge:
    source: str
    target: str


@dataclass(frozen=True)
class DependencyGraphNormalization:
    raw_edges: list[DependencyEdge]
    normalized_edges: list[DependencyEdge]
    cycles: list[list[str]]
    claims_in_cycles: frozenset[str]
    self_loop_count: int
    removed_internal_edges: list[DependencyEdge]
    normalized_depends_on: dict[str, list[str]]
    cycle_membership: dict[str, bool]


def build_adjacency(edges: list[DependencyEdge]) -> dict[str, list[str]]:
    adjacency: dict[str, list[str]] = {}
    for edge in edges:
        adjacency.setdefault(edge.source, [])
        adjacency.setdefault(edge.target, [])
        if edge.target not in adjacency[edge.source]:
            adjacency[edge.source].append(edge.target)
    return adjacency


def validate_raw_dependency_edges(
    edges: list[DependencyEdge],
    *,
    known_claim_ids: frozenset[str],
    allowed_premise_ids: frozenset[str] | None = None,
) -> None:
    """Validate raw edges. Cycles are allowed; unknown/duplicate/illegal premises are not."""
    seen: set[tuple[str, str]] = set()
    for edge in edges:
        if edge.source not in known_claim_ids:
            msg = f"dependency edge source unknown claim_id {edge.source!r}"
            raise ValueError(msg)
        if edge.target not in known_claim_ids:
            msg = f"dependency edge target unknown claim_id {edge.target!r}"
            raise ValueError(msg)
        key = (edge.source, edge.target)
        if key in seen:
            msg = f"duplicate raw dependency edge {edge.source} -> {edge.target}"
            raise ValueError(msg)
        seen.add(key)
        if allowed_premise_ids is not None and edge.target not in allowed_premise_ids:
            msg = (
                f"dependency {edge.source} -> {edge.target} targets a claim "
                "outside the Stage 2B premise pool"
            )
            raise ValueError(msg)


def detect_cycles(adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Return SCCs that are cyclic (size > 1 or a self-loop), sorted deterministically."""
    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[list[str]] = []

    nodes = sorted(adjacency.keys())

    def strongconnect(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for successor in adjacency.get(node, []):
            if successor not in indices:
                strongconnect(successor)
                lowlinks[node] = min(lowlinks[node], lowlinks[successor])
            elif successor in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[successor])

        if lowlinks[node] == indices[node]:
            component: list[str] = []
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.append(member)
                if member == node:
                    break
            components.append(sorted(component))

    for node in nodes:
        if node not in indices:
            strongconnect(node)

    cycles: list[list[str]] = []
    for component in components:
        if len(component) > 1:
            cycles.append(component)
            continue
        alone = component[0]
        if alone in adjacency.get(alone, []):
            cycles.append(component)

    def _cycle_sort_key(group: list[str]) -> tuple[int, str]:
        first = group[0] if group else ""
        return (len(group), first)

    cycles.sort(key=_cycle_sort_key)
    return cycles


def normalize_dependency_graph(
    edges: list[DependencyEdge],
) -> DependencyGraphNormalization:
    adjacency = build_adjacency(edges)
    cycles = detect_cycles(adjacency)
    claims_in_cycles = frozenset(claim_id for group in cycles for claim_id in group)
    cycle_membership = {node: node in claims_in_cycles for node in adjacency}

    self_loop_count = sum(1 for edge in edges if edge.source == edge.target)
    removed_internal: list[DependencyEdge] = []
    normalized: list[DependencyEdge] = []

    for edge in edges:
        both_in_same_cycle = False
        if edge.source in claims_in_cycles and edge.target in claims_in_cycles:
            for group in cycles:
                group_set = set(group)
                if edge.source in group_set and edge.target in group_set:
                    both_in_same_cycle = True
                    break
        if both_in_same_cycle:
            removed_internal.append(edge)
            continue
        normalized.append(edge)

    normalized_depends_on: dict[str, list[str]] = {node: [] for node in adjacency}
    for edge in normalized:
        if edge.target not in normalized_depends_on[edge.source]:
            normalized_depends_on[edge.source].append(edge.target)
    for node in normalized_depends_on:
        normalized_depends_on[node] = sorted(normalized_depends_on[node])

    return DependencyGraphNormalization(
        raw_edges=list(edges),
        normalized_edges=normalized,
        cycles=cycles,
        claims_in_cycles=claims_in_cycles,
        self_loop_count=self_loop_count,
        removed_internal_edges=removed_internal,
        normalized_depends_on=normalized_depends_on,
        cycle_membership=cycle_membership,
    )


def edges_from_dependency_map(depends_on_by_claim: dict[str, list[str]]) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    for source in sorted(depends_on_by_claim.keys()):
        for target in depends_on_by_claim[source]:
            edges.append(DependencyEdge(source=source, target=target))
    return edges
