"""Graph configs and classes for the configuration model family."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from craeft.graphs.base import GraphConfig, UndirectedGraph
from craeft.graphs.configuration_model.connection import Connector
from craeft.graphs.configuration_model.sequence import SubgraphSequence
from craeft.graphs.metrics.clustering import global_clustering_coefficient

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
        connector = Connector(config.n, rng)
        connector.connect_singles(config.degrees)
        return cls(connector.to_csr())

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
