"""Erdos-Renyi random graph G(n, p)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from scipy.sparse import coo_matrix, csr_matrix

from craeft.graphs.base import GraphConfig, UndirectedGraph
from craeft.graphs.metrics.clustering import (
    global_clustering_coefficient,
)


def random_graph(
    n: int, p: float, rng: np.random.Generator | None = None
) -> csr_matrix:
    """Generate an Erdos-Renyi random graph G(n, p).

    Args:
        n: Number of nodes.
        p: Probability of each edge existing.
        rng: Optional random generator for reproducibility.

    Returns:
        Symmetric adjacency matrix in CSR format (no self-loops).
    """
    rng = rng or np.random.default_rng()

    # Number of possible edges in upper triangle (no self-loops)
    max_edges = n * (n - 1) // 2

    # Sample how many edges, then which ones
    num_edges = rng.binomial(max_edges, p)

    if num_edges == 0:
        return csr_matrix((n, n), dtype=np.int8)

    flat_indices = rng.choice(max_edges, size=num_edges, replace=False)

    # Decode flat index to (i, j) upper-triangle coordinates
    i = ((np.sqrt(1 + 8 * flat_indices) - 1) // 2).astype(np.int64) + 1
    j = flat_indices - i * (i - 1) // 2

    # Build symmetric adjacency
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.ones(2 * num_edges, dtype=np.int8)

    return coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8).tocsr()


@dataclass(frozen=True)
class ErdosRenyiConfig(GraphConfig):
    """Configuration for an Erdos-Renyi random graph.

    Attributes:
        n: Number of nodes.
        p: Probability of each edge existing, in [0, 1].
    """

    p: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.p <= 1.0:
            msg = f"p must be in [0, 1], got {self.p}"
            raise ValueError(msg)


class ErdosRenyiGraph(UndirectedGraph[ErdosRenyiConfig]):
    """Erdos-Renyi random graph G(n, p).

    Each possible edge exists independently with probability p.
    """

    @classmethod
    def from_config(
        cls,
        config: ErdosRenyiConfig,
        rng: np.random.Generator,
    ) -> Self:
        """Generate an Erdos-Renyi graph.

        Args:
            config: Graph configuration (n, p).
            rng: Random number generator.

        Returns:
            An ErdosRenyiGraph instance.
        """
        adjacency = random_graph(config.n, config.p, rng)
        return cls(adjacency)

    @property
    def clustering_coefficient(self) -> float:
        return global_clustering_coefficient(self._adjacency)
