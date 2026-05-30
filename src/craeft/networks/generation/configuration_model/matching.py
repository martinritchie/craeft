"""Greedy cardinality matching algorithm.

The core of the CMA: assigns unordered hyperstub tuples to specific
nodes by descending induced degree, ensuring no node is over-allocated
or assigned to the same subgraph twice.

Reference:
    Ritchie et al. (2015) "Generation and analysis of networks with a
    prescribed degree sequence and subgraph family", Section 2.2,
    Algorithm 2 (page 29).
"""

import numpy as np
from numpy.typing import NDArray

from .hyperstub_matrix import HyperstubAllocation
from .subgraph_spec import SubgraphSpec


def cardinality_match(
    degrees: NDArray[np.int_],
    specs: list[SubgraphSpec],
    decompositions: list[dict[int, NDArray[np.int_]]],
    rng: np.random.Generator,
) -> HyperstubAllocation:
    """Assign hyperstub tuples to nodes via greedy cardinality matching.

    For each subgraph spec, computes the induced degree (stub cost) of
    each hyperstub tuple, then greedily assigns tuples to eligible nodes
    in descending order of cost. Nodes must have sufficient remaining
    degree and not already be assigned to that subgraph.

    Args:
        degrees: Per-node degree sequence.
        specs: Subgraph specifications.
        decompositions: Per-spec multinomial decompositions (unordered).
        rng: Random generator for tie-breaking.

    Returns:
        HyperstubAllocation with node-assigned bins and remaining singles.
    """
    n = len(degrees)
    remaining = degrees.copy()

    # Per-spec, per-node allocation tracking
    assigned: list[dict[int, NDArray[np.int_]]] = []
    used_per_spec: list[set[int]] = []

    for spec, decomp in zip(specs, decompositions):
        node_alloc = {t: np.zeros(n, dtype=np.int_) for t in decomp}
        assigned.append(node_alloc)
        used_per_spec.append(set())

    # Build list of (spec_index, induced_cost, corner_type_counts)
    tuples_to_assign = _build_tuples(specs, decompositions)

    # Sort descending by induced cost
    tuples_to_assign.sort(key=lambda x: x[1], reverse=True)

    for spec_idx, cost, type_counts in tuples_to_assign:
        # Find eligible nodes: enough remaining degree and not yet used
        eligible = np.where(remaining >= cost)[0]
        eligible = np.array(
            [i for i in eligible if i not in used_per_spec[spec_idx]],
            dtype=np.int_,
        )

        if len(eligible) == 0:
            # Shed back to singles — can't place this tuple
            continue

        # Randomly pick one eligible node
        node = int(rng.choice(eligible))
        remaining[node] -= cost
        used_per_spec[spec_idx].add(node)

        # Record the allocation
        for corner_type, count in type_counts.items():
            assigned[spec_idx][corner_type][node] += count

    # Build the allocation
    bins: dict[tuple[int, int], NDArray[np.int_]] = {}
    for i, node_alloc in enumerate(assigned):
        for corner_type, counts in node_alloc.items():
            bins[(i, corner_type)] = counts

    return HyperstubAllocation(
        bins=bins,
        singles=remaining,
        specs=tuple(specs),
    )


def _build_tuples(
    specs: list[SubgraphSpec],
    decompositions: list[dict[int, NDArray[np.int_]]],
) -> list[tuple[int, int, dict[int, int]]]:
    """Build the list of hyperstub tuples to assign.

    Each tuple represents one node's participation in one subgraph,
    with its induced stub cost and corner-type breakdown.
    """
    tuples: list[tuple[int, int, dict[int, int]]] = []

    for spec_idx, (spec, decomp) in enumerate(zip(specs, decompositions)):
        motif = spec.motif
        # For each node with non-zero allocation in this spec
        n = len(spec.sequence)
        for node in range(n):
            type_counts = {t: int(decomp[t][node]) for t in decomp}
            if all(v == 0 for v in type_counts.values()):
                continue
            cost = sum(
                count * motif.cardinalities[t] for t, count in type_counts.items()
            )
            tuples.append((spec_idx, cost, type_counts))

    return tuples
