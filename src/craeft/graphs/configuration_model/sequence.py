"""Sequence types, sampling, and allocation for the configuration model family.

Owns the full sequence lifecycle:
    - Types: SubgraphSequence, Allocation
    - Sampling: degree sequences, subgraph participation sequences
    - Matching: greedy allocation of hyperstubs to nodes
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_discrete

from craeft.graphs.base import Subgraph

# ---------------------------------------------------------------------------
# SubgraphSequence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphSequence:
    """A subgraph paired with a participation distribution.

    Describes which subgraph structure to embed and how participation
    counts are distributed across nodes. The concrete sequence is
    sampled at generation time via ``sample``.

    CMA-specific properties (orbits, orbit degrees) are derived
    from the subgraph's automorphism group.

    Attributes:
        subgraph: The subgraph structure to embed.
        distribution: Frozen scipy discrete distribution for
            per-node participation counts (e.g. poisson(1)).
    """

    subgraph: Subgraph
    distribution: rv_discrete

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.int_]:
        """Sample a participation sequence of length n.

        Rejection samples until all values are in [0, n-1] and the
        total is divisible by the subgraph's node count.

        Args:
            n: Number of nodes in the network.
            rng: Random number generator.

        Returns:
            Array of n non-negative counts whose sum is divisible
            by subgraph.num_nodes.
        """
        return _sample_sequence(
            n, self.distribution, rng, divisor=self.subgraph.num_nodes
        )

    def _decompose(
        self,
        sequence: NDArray[np.int_],
        rng: np.random.Generator,
    ) -> dict[int, NDArray[np.int_]]:
        """Decompose a sampled sequence into per-orbit counts.

        For vertex-transitive subgraphs (single orbit), returns the
        sequence unchanged. For non-transitive subgraphs, uses the
        multinomial distribution and rejects until column totals
        match the exact orbit proportions.

        Args:
            sequence: A sampled participation sequence from ``sample``.
            rng: Random number generator.

        Returns:
            Mapping from orbit label to per-node count array.
        """
        ...

    @property
    def orbits(self) -> list[int]:
        """Orbit label for each node in the subgraph.

        Nodes in the same automorphism orbit receive the same label.
        """
        ...

    @property
    def orbit_degrees(self) -> dict[int, int]:
        """Maps orbit label to its degree within the subgraph."""
        ...

    @property
    def orbit_sizes(self) -> dict[int, int]:
        """Maps orbit label to how many nodes belong to that orbit."""
        ...

    @property
    def is_vertex_transitive(self) -> bool:
        """True if all nodes belong to a single orbit."""
        ...

    def edges_for(self, nodes: NDArray[np.int_]) -> tuple[list[int], list[int]]:
        """Map the subgraph's structure onto concrete node IDs."""
        ...


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_degree_sequence(
    n: int,
    distribution: rv_discrete,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Sample a valid degree sequence from a discrete distribution.

    Draws n degrees independently and rejects sequences where the
    sum is odd or any degree exceeds n - 1. Expected few attempts
    for well-chosen distributions.

    Args:
        n: Number of nodes.
        distribution: Frozen scipy discrete distribution (e.g. poisson(3)).
        rng: Random number generator.

    Returns:
        Array of n non-negative degrees with even sum,
        each at most n - 1.
    """
    return _sample_sequence(n, distribution, rng, divisor=2)


def _sample_sequence(
    n: int,
    distribution: rv_discrete,
    rng: np.random.Generator,
    divisor: int,
) -> NDArray[np.int_]:
    """Sample a sequence with bounded values and divisible sum.

    Rejection samples until all values are in [0, n-1] and the
    sum is divisible by divisor.
    """
    max_value = n - 1

    while True:
        values = distribution.rvs(size=n, random_state=rng)
        if values.min() < 0:
            continue
        if values.max() > max_value:
            continue
        if int(values.sum()) % divisor != 0:
            continue
        return values


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Allocation:
    """Result of greedy matching: hyperstub bins and remaining singles.

    Attributes:
        bins: Mapping from (sequence_index, orbit) to per-node
            hyperstub counts.
        singles: Per-node count of remaining single stubs.
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


def cardinality_match(
    degrees: NDArray[np.int_],
    sequences: list[SubgraphSequence],
    decompositions: list[dict[int, NDArray[np.int_]]],
    rng: np.random.Generator,
) -> Allocation:
    """Assign hyperstub tuples to nodes via greedy cardinality matching.

    Computes the induced degree (stub cost) of each hyperstub tuple,
    then greedily assigns tuples to eligible nodes in descending
    order of cost. Nodes must have sufficient remaining degree and
    not already be assigned to that subgraph.

    If a tuple cannot be placed, raises instead of silently shedding.

    Args:
        degrees: Per-node degree sequence.
        sequences: Subgraph sequences.
        decompositions: Per-sequence multinomial decompositions
            (orbit label to per-node count arrays).
        rng: Random generator for tie-breaking.

    Returns:
        Allocation with node-assigned bins and remaining singles.

    Raises:
        AllocationError: If a hyperstub tuple cannot be placed.
    """
    ...


class AllocationError(Exception):
    """Raised when the greedy matching cannot place a hyperstub tuple."""
