"""Graph config and class for the configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.graphs.base import GraphConfig, UndirectedGraph
from craeft.graphs.configuration_model.connection import (
    ConnectionError,
    Connector,
)
from craeft.graphs.configuration_model.sequence import (
    AllocationError,
    SubgraphSequence,
    allocate_subgraphs,
)
from craeft.graphs.metrics.clustering import global_clustering_coefficient
from craeft.graphs.metrics.subgraph import designed_clustering as _designed_clustering


@dataclass(frozen=True)
class ConfigModelConfig(GraphConfig):
    """Configuration for a configuration model graph.

    Attributes:
        n: Number of nodes.
        degrees: Per-node degree sequence. Length must equal n,
            values non-negative, sum must be even.
        sequences: Subgraph sequences to embed. When empty, produces
            a standard configuration model graph (edge-only).
        max_retries: Maximum reset-and-retry attempts when subgraph
            allocation or connection fails.
    """

    degrees: NDArray[np.int_]
    sequences: tuple[SubgraphSequence, ...] = ()
    max_retries: int = 100

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
    """Random graph with a prescribed degree sequence and optional
    subgraph structure.

    When no subgraph sequences are specified, generates a standard
    configuration model graph via stub pairing. When sequences are
    provided, splits them by orbit, greedily allocates participations
    to nodes, connects subgraph instances (checking for duplicates
    and existing edges), then pairs remaining single stubs.

    Retries from scratch on failure (dead-end configurations).
    """

    def __init__(
        self,
        adjacency: csr_matrix,
        config: ConfigModelConfig | None = None,
    ) -> None:
        super().__init__(adjacency)
        self._config = config

    @classmethod
    def from_config(
        cls,
        config: ConfigModelConfig,
        rng: np.random.Generator,
    ) -> Self:
        """Generate a configuration model graph.

        Without subgraph sequences:
            1. Pair stubs from the degree sequence
            2. Remove self-loops and multi-edges

        With subgraph sequences:
            1. Sample participation sequences
            2. Split each by orbit
            3. Greedily allocate to nodes
            4. Connect subgraph instances (with multi-edge check)
            5. Pair remaining single stubs
            6. Assemble into adjacency matrix

        On failure (AllocationError or ConnectionError), retries
        up to config.max_retries times with fresh random state.

        Args:
            config: Configuration model configuration.
            rng: Random number generator.

        Returns:
            A ConfigModelGraph instance.

        Raises:
            RuntimeError: If all retries exhausted.
        """
        if not config.sequences:
            connector = Connector(config.n, rng)
            connector.connect_singles(config.degrees)
            return cls(connector.to_csr(), config=config)

        # Subgraph sequence pipeline
        for _ in range(config.max_retries):
            try:
                connector = Connector(config.n, rng)
                decompositions: list[dict[int, NDArray[np.int_]]] = []

                # 1. Sample participation sequences and split by orbit
                for seq in config.sequences:
                    parts = seq.sample(config.n, rng)
                    decomp = seq._split_by_orbit(parts, rng)
                    decompositions.append(decomp)

                # 2. Allocate subgraphs to nodes (verify degree budget)
                allocation = allocate_subgraphs(
                    degrees=config.degrees,
                    sequences=list(config.sequences),
                    decompositions=decompositions,
                    rng=rng,
                )

                # 3. Connect subgraph instances
                for seq_idx in range(len(config.sequences)):
                    connector.connect_subgraph(
                        sequence=config.sequences[seq_idx],
                        allocation=allocation,
                        sequence_index=seq_idx,
                    )

                # 4. Pair remaining single stubs
                connector.connect_singles(allocation.singles)

                # 5. Assemble into adjacency matrix
                return cls(connector.to_csr(), config=config)

            except (AllocationError, ConnectionError, ValueError, RuntimeError):
                # Retry with a fresh random state on any failure
                continue

        msg = (
            f"Failed to generate graph after {config.max_retries} "
            f"retries"
        )
        raise RuntimeError(msg)

    @property
    def clustering_coefficient(self) -> float:
        return global_clustering_coefficient(self._adjacency)

    @property
    def designed_clustering(self) -> float:
        """Designed clustering coefficient implied by the originating config.

        The closed-form value computed from the config before generation
        (see `craeft.graphs.metrics.designed_clustering`), for comparison
        against the realized `clustering_coefficient`. The realized value
        will typically exceed this by a small by-product term (random
        closure from stub pairing plus subgraph overlap).

        Raises:
            ValueError: If the graph was not built via `from_config` (so
                has no attached config to compute the designed value from).
        """
        if self._config is None:
            msg = "designed_clustering requires a graph built via from_config"
            raise ValueError(msg)
        return _designed_clustering(self._config)
