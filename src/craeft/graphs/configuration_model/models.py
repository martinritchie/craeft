"""Data models and graph classes for the configuration model family.

Contains all user-facing types: configs, graph classes, and the
subgraph specification for CMA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from craeft.graphs.base import GraphConfig, Subgraph, UndirectedGraph
from craeft.graphs.configuration_model.connection import connect_singles
from craeft.graphs.metrics.clustering import global_clustering_coefficient

# ---------------------------------------------------------------------------
# SubgraphSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubgraphSpec:
    """A subgraph paired with per-node participation counts.

    Derives CMA-specific properties (corner types, cardinalities)
    from the subgraph's degree structure.

    Attributes:
        subgraph: The subgraph structure to embed.
        sequence: Per-node count of subgraph instances. Length must
            equal the number of nodes in the network.
    """

    subgraph: Subgraph
    sequence: NDArray[np.int_]

    def __post_init__(self) -> None: ...

    @property
    def total(self) -> int:
        """Total subgraph instances across all nodes."""
        ...

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the network (length of sequence)."""
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
        specs: Subgraph specifications pairing subgraphs with
            per-node participation sequences.
        max_retries: Maximum reset-and-retry attempts.
    """

    degrees: NDArray[np.int_]
    specs: tuple[SubgraphSpec, ...]
    max_retries: int = 100

    def __post_init__(self) -> None: ...


class CMAGraph(UndirectedGraph[CMAConfig]):
    """Network with prescribed degree sequence and subgraph structure.

    Generated via the Cardinality Matching Algorithm: decomposes
    subgraph sequences, greedily matches hyperstubs to nodes,
    connects subgraph instances (checking for duplicates and
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
            1. Multinomial decomposition per subgraph spec
            2. Greedy cardinality matching
            3. Connect subgraph instances (with multi-edge check)
            4. Pair remaining single stubs
            5. Assemble into adjacency matrix

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
