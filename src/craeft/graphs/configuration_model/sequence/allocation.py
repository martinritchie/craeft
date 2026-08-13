"""Greedy subgraph allocation to nodes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from craeft.graphs.configuration_model.sequence.subgraph_sequence import (
    SubgraphSequence,
)


@dataclass(frozen=True)
class Allocation:
    """Result of greedy subgraph allocation.

    Attributes:
        bins: Mapping from (sequence_index, orbit) to per-node
            participation counts for that orbit.
        singles: Per-node count of remaining single stubs after
            subgraph allocations are subtracted from the degree
            sequence.
    """

    bins: dict[tuple[int, int], NDArray[np.int_]]
    singles: NDArray[np.int_]

    def node_ids_for(self, sequence_index: int, orbit: int) -> NDArray[np.int_]:
        """Flat array of node IDs for a specific bin.

        Each node i appears bins[(sequence_index, orbit)][i] times.
        """
        counts = self.bins[(sequence_index, orbit)]
        return np.repeat(np.arange(len(counts)), counts).astype(np.int_)

    def single_stubs(self) -> NDArray[np.int_]:
        """Flat array of node IDs for remaining single stubs."""
        return np.repeat(np.arange(len(self.singles)), self.singles).astype(np.int_)


def allocate_subgraphs(
    degrees: NDArray[np.int_],
    sequences: list[SubgraphSequence],
    decompositions: list[dict[int, NDArray[np.int_]]],
    rng: np.random.Generator,
) -> Allocation:
    """Verify subgraph participations fit within the degree budget.

    Each decomposition (from ``SubgraphSequence._split_by_orbit``)
    already specifies per-node per-orbit counts. This function
    verifies that the total stub cost per node — summing over all
    sequences and orbits (count × orbit_degree) — does not exceed
    the node's degree budget.

    On success, builds the ``Allocation`` bins (mapping each
    (sequence_index, orbit) to its per-node count) and computes
    remaining single stubs.

    Args:
        degrees: Per-node degree sequence.
        sequences: Subgraph sequences (for orbit degree lookup).
        decompositions: Per-sequence orbit splits (orbit label
            to per-node count arrays).
        rng: Random generator (unused; accepted for API symmetry).

    Returns:
        Allocation with node-assigned bins and remaining singles.

    Raises:
        AllocationError: If any node's stub cost exceeds its degree.
    """
    n = len(degrees)
    total_cost = np.zeros(n, dtype=np.int_)
    bins: dict[tuple[int, int], NDArray[np.int_]] = {}

    for seq_idx, (seq, decomp) in enumerate(zip(sequences, decompositions)):
        for orbit, counts in decomp.items():
            # Verify this orbit's counts are non-negative
            if np.any(counts < 0):
                msg = (
                    f"Negative counts in sequence {seq_idx}, "
                    f"orbit {orbit}"
                )
                raise AllocationError(msg)
            # Accumulate stub cost
            cost = counts * seq.orbit_degrees[orbit]
            total_cost += cost
            bins[(seq_idx, orbit)] = counts

    # Verify every node can afford its total subgraph cost
    if np.any(total_cost > degrees):
        over = np.where(total_cost > degrees)[0]
        msg = (
            f"Degree budget exceeded for {len(over)} node(s). "
            f"First violation at node {over[0]}: "
            f"cost={int(total_cost[over[0]])} > "
            f"degree={int(degrees[over[0]])}"
        )
        raise AllocationError(msg)

    singles = degrees - total_cost

    return Allocation(bins=bins, singles=singles)


class AllocationError(Exception):
    """Raised when a subgraph participation cannot be assigned to any node."""
