"""Configuration model random graph with prescribed degree sequence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

from craeft.graphs.base import GraphConfig, UndirectedGraph
from craeft.graphs.configuration_model.pairing import pair_stubs
from craeft.graphs.metrics.clustering import global_clustering_coefficient


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
        adjacency = pair_stubs(config.degrees, rng)
        return cls(adjacency)

    @property
    def clustering_coefficient(self) -> float:
        return global_clustering_coefficient(self._adjacency)
