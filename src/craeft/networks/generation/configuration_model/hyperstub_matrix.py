"""Aggregated hyperstub allocation across multiple subgraph families.

Collects corner-type allocations from all subgraph specs and tracks
remaining single stubs per node.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .multinomial_partition import compute_stub_cost
from .subgraph_spec import SubgraphSpec


@dataclass(frozen=True)
class HyperstubAllocation:
    """Result of allocating hyperstubs across multiple subgraph families.

    Attributes:
        bins: Mapping from (spec_index, corner_type) to per-node
            hyperstub counts for that corner type in that subgraph.
        singles: Per-node count of remaining stubs for single pairing.
        specs: The subgraph specs used in the allocation.
    """

    bins: dict[tuple[int, int], NDArray[np.int_]]
    singles: NDArray[np.int_]
    specs: tuple[SubgraphSpec, ...]

    def node_ids_for(
        self,
        spec_index: int,
        corner_type: int,
    ) -> NDArray[np.int_]:
        """Build a flat array of node IDs for a specific bin.

        Each node i appears bins[(spec_index, corner_type)][i] times.
        """
        counts = self.bins[(spec_index, corner_type)]
        return np.repeat(np.arange(len(counts)), counts)

    def single_stubs(self) -> NDArray[np.int_]:
        """Build a flat array of node IDs for single stubs."""
        return np.repeat(np.arange(len(self.singles)), self.singles)


def build_allocation(
    degrees: NDArray[np.int_],
    specs: list[SubgraphSpec],
    decompositions: list[dict[int, NDArray[np.int_]]],
) -> HyperstubAllocation:
    """Build a hyperstub allocation from degree sequence and decompositions.

    Subtracts the stub cost of all subgraph allocations from the degree
    sequence to compute remaining singles.

    Args:
        degrees: Per-node degree sequence.
        specs: Subgraph specifications.
        decompositions: Per-spec multinomial decompositions.

    Returns:
        HyperstubAllocation with bins and singles.

    Raises:
        ValueError: If total stub cost exceeds any node's degree.
    """
    n = len(degrees)
    total_cost = np.zeros(n, dtype=np.int_)

    bins: dict[tuple[int, int], NDArray[np.int_]] = {}
    for i, (spec, decomp) in enumerate(zip(specs, decompositions)):
        cost = compute_stub_cost(decomp, spec.motif)
        total_cost += cost
        for corner_type, counts in decomp.items():
            bins[(i, corner_type)] = counts

    if np.any(total_cost > degrees):
        over = np.where(total_cost > degrees)[0]
        msg = (
            f"Stub cost exceeds degree at nodes {over[:5].tolist()}"
            f" (cost={total_cost[over[0]]}, degree={degrees[over[0]]})"
        )
        raise ValueError(msg)

    singles = degrees - total_cost

    return HyperstubAllocation(
        bins=bins,
        singles=singles,
        specs=tuple(specs),
    )
