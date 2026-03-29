"""Sequence sampling, transformation, and allocation for the configuration model family.

Handles the full sequence lifecycle:
    - Sampling: degree sequences, subgraph participation sequences
    - Decomposition: multinomial split into corner-type counts
    - Matching: greedy allocation of hyperstubs to nodes
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_discrete

if TYPE_CHECKING:
    from craeft.graphs.configuration_model.models import SubgraphSpec


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


def sample_subgraph_sequence(
    n: int,
    distribution: rv_discrete,
    subgraph_nodes: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Sample a per-node subgraph participation sequence.

    Draws n participation counts independently and rejects
    sequences where the total is not divisible by the number
    of nodes in the subgraph (required to form complete instances).

    Args:
        n: Number of nodes.
        distribution: Frozen scipy discrete distribution for
            per-node participation counts (e.g. poisson(1)).
        subgraph_nodes: Number of nodes in the subgraph. The total
            participation count must be divisible by this.
        rng: Random number generator.

    Returns:
        Array of n non-negative counts, each at most n - 1,
        whose sum is divisible by subgraph_nodes.
    """
    return _sample_sequence(n, distribution, rng, divisor=subgraph_nodes)


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


def decompose(
    spec: SubgraphSpec,
    rng: np.random.Generator,
) -> dict[int, NDArray[np.int_]]:
    """Decompose a subgraph sequence into per-corner-type counts.

    For complete subgraphs (single corner type), returns the
    sequence unchanged. For incomplete subgraphs, uses the
    multinomial distribution and rejects until column totals
    match the exact corner-type proportions.

    Args:
        spec: Subgraph specification with subgraph and sequence.
        rng: Random number generator.

    Returns:
        Mapping from corner type to per-node count array.
    """
    ...


# ---------------------------------------------------------------------------
# Matching
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Allocation:
    """Result of greedy matching: hyperstub bins and remaining singles.

    Attributes:
        bins: Mapping from (spec_index, corner_type) to per-node
            hyperstub counts.
        singles: Per-node count of remaining single stubs.
    """

    bins: dict[tuple[int, int], NDArray[np.int_]]
    singles: NDArray[np.int_]

    def node_ids_for(self, spec_index: int, corner_type: int) -> NDArray[np.int_]:
        """Flat array of node IDs for a specific bin.

        Each node i appears bins[(spec_index, corner_type)][i] times.
        """
        ...

    def single_stubs(self) -> NDArray[np.int_]:
        """Flat array of node IDs for remaining single stubs."""
        ...


def cardinality_match(
    degrees: NDArray[np.int_],
    specs: list[SubgraphSpec],
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
        specs: Subgraph specifications.
        decompositions: Per-spec multinomial decompositions.
        rng: Random generator for tie-breaking.

    Returns:
        Allocation with node-assigned bins and remaining singles.

    Raises:
        AllocationError: If a hyperstub tuple cannot be placed.
    """
    ...


class AllocationError(Exception):
    """Raised when the greedy matching cannot place a hyperstub tuple."""
