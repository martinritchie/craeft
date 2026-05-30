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
        ...

    def single_stubs(self) -> NDArray[np.int_]:
        """Flat array of node IDs for remaining single stubs."""
        ...


def allocate_subgraphs(
    degrees: NDArray[np.int_],
    sequences: list[SubgraphSequence],
    decompositions: list[dict[int, NDArray[np.int_]]],
    rng: np.random.Generator,
) -> Allocation:
    """Greedily assign subgraph participations to nodes.

    Computes the stub cost of each node's orbit counts, then
    assigns them to eligible nodes in descending order of cost.
    Nodes must have sufficient remaining degree and not already
    be assigned to that subgraph.

    If a participation cannot be placed, raises instead of
    silently shedding.

    Args:
        degrees: Per-node degree sequence.
        sequences: Subgraph sequences.
        decompositions: Per-sequence orbit splits (orbit label
            to per-node count arrays).
        rng: Random generator for tie-breaking.

    Returns:
        Allocation with node-assigned bins and remaining singles.

    Raises:
        AllocationError: If a participation cannot be assigned
            to any node.
    """
    ...


class AllocationError(Exception):
    """Raised when a subgraph participation cannot be assigned to any node."""
