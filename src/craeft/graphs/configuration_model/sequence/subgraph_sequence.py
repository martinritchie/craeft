"""SubgraphSequence: subgraph paired with a participation distribution."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_discrete

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model.sequence.sampling import (
    _sample_sequence,
)


@dataclass(frozen=True)
class SubgraphSequence:
    """A subgraph paired with a participation distribution.

    Describes which subgraph structure to embed and how participation
    counts are distributed across nodes. The concrete sequence is
    sampled at generation time via ``sample``.

    Orbit properties are derived from the subgraph's automorphism
    group.

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

    def _split_by_orbit(
        self,
        sequence: NDArray[np.int_],
        rng: np.random.Generator,
    ) -> dict[int, NDArray[np.int_]]:
        """Split a participation sequence into per-orbit counts.

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
