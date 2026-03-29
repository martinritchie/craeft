"""Data models and graph classes for the configuration model family.

Contains all user-facing types: configs, graph classes, and the
subgraph sequence for CMA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_discrete

from craeft.graphs.base import GraphConfig, Subgraph, UndirectedGraph
from craeft.graphs.configuration_model.connection import connect_singles
from craeft.graphs.metrics.clustering import global_clustering_coefficient

# ---------------------------------------------------------------------------
# Decomposition result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Decomposition:
    """Per-corner-type counts from multinomial decomposition.

    Attributes:
        counts: Mapping from corner type to per-node count array.
    """

    counts: dict[int, NDArray[np.int_]]


# ---------------------------------------------------------------------------
# SubgraphSequence
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphSequence:
    """A subgraph paired with a participation distribution.

    Describes which subgraph structure to embed and how participation
    counts are distributed across nodes. The concrete sequence is
    sampled at generation time via ``sample``.

    CMA-specific properties (corner types, cardinalities) are derived
    from the subgraph's degree structure.

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
        from craeft.graphs.configuration_model.sequence import (
            _sample_sequence,
        )

        return _sample_sequence(
            n, self.distribution, rng, divisor=self.subgraph.num_nodes
        )

    def _decompose(
        self,
        sequence: NDArray[np.int_],
        rng: np.random.Generator,
    ) -> Decomposition:
        """Decompose a sampled sequence into corner-type counts.

        For complete subgraphs (single corner type), returns the
        sequence unchanged. For incomplete subgraphs, uses the
        multinomial distribution and rejects until column totals
        match the exact corner-type proportions.

        Args:
            sequence: A sampled participation sequence from ``sample``.
            rng: Random number generator.

        Returns:
            Decomposition with per-corner-type count arrays.
        """
        ...

    @property
    def corner_types(self) -> list[int]:
        """Corner type for each node in the subgraph (derived from degree)."""
        ...

    @property
    def cardinalities(self) -> dict[int, int]:
        """Maps corner type to its degree within the subgraph."""
        ...

    @property
    def type_counts(self) -> dict[int, int]:
        """Maps corner type to how many nodes have that type."""
        ...

    @property
    def is_complete(self) -> bool:
        """True if all nodes in the subgraph have equal degree."""
        ...

    def edges_for(self, nodes: NDArray[np.int_]) -> tuple[list[int], list[int]]:
        """Map the subgraph's structure onto concrete node IDs."""
        ...


# ---------------------------------------------------------------------------
# Vanilla configuration model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ConfigModelConfig(GraphConfig):
    """Configuration for a configuration model graph.

    Attributes:
        n: Number of nodes.
        degrees: Per-node degree sequence. Length must equal n,
            values non-negative, sum must be even.
    """

    degrees: NDArray[np.int_]

    def __post_init__(self) -> None:
        if len(self.degrees) != self.n:
            msg = f"Degree sequence length {len(self.degrees)} != n ({self.n})"
            raise ValueError(msg)
        if np.any(self.degrees < 0):
            msg = "Degrees must be non-negative"
            raise ValueError(msg)
        if int(self.degrees.sum()) % 2 != 0:
            msg = f"Degree sum must be even, got {int(self.degrees.sum())}"
            raise ValueError(msg)


class ConfigModelGraph(UndirectedGraph[ConfigModelConfig]):
    """Random graph with a prescribed degree sequence.

    Generated via stub pairing: each node contributes stubs equal
    to its degree, stubs are randomly paired to form edges.
    Self-loops and multi-edges are removed.
    """

    @classmethod
    def from_config(
        cls,
        config: ConfigModelConfig,
        rng: np.random.Generator,
    ) -> Self:
        adjacency = connect_singles(config.degrees, rng)
        return cls(adjacency)

    @property
    def clustering_coefficient(self) -> float:
        return global_clustering_coefficient(self._adjacency)


# ---------------------------------------------------------------------------
# CMA (Cardinality Matching Algorithm)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CMAConfig(GraphConfig):
    """Configuration for a CMA-generated graph.

    Attributes:
        n: Number of nodes.
        degrees: Per-node degree sequence.
        sequences: Subgraph sequences defining which subgraphs
            to embed and their participation distributions.
        max_retries: Maximum reset-and-retry attempts.
    """

    degrees: NDArray[np.int_]
    sequences: tuple[SubgraphSequence, ...]
    max_retries: int = 100

    def __post_init__(self) -> None: ...


class CMAGraph(UndirectedGraph[CMAConfig]):
    """Network with prescribed degree sequence and subgraph structure.

    Generated via the Cardinality Matching Algorithm: samples and
    decomposes subgraph sequences, greedily matches hyperstubs to
    nodes, connects subgraph instances (checking for duplicates and
    existing edges), then pairs remaining single stubs.

    Retries from scratch on failure (dead-end configurations).
    """

    @classmethod
    def from_config(
        cls,
        config: CMAConfig,
        rng: np.random.Generator,
    ) -> Self:
        """Generate a CMA graph.

        The algorithm:
            1. Sample participation sequences from each SubgraphSequence
            2. Multinomial decomposition per sequence
            3. Greedy cardinality matching
            4. Connect subgraph instances (with multi-edge check)
            5. Pair remaining single stubs
            6. Assemble into adjacency matrix

        On failure (AllocationError or ConnectionError), retries
        up to config.max_retries times with fresh random state.

        Args:
            config: CMA configuration.
            rng: Random number generator.

        Returns:
            A CMAGraph instance.

        Raises:
            RuntimeError: If all retries exhausted.
        """
        ...

    @property
    def clustering_coefficient(self) -> float: ...
