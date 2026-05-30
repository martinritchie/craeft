"""Network generator protocol and concrete implementations.

Defines a structural typing contract for network generation and provides
frozen dataclass implementations for each generation algorithm. All
generators are stateless and picklable for multiprocessing.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


class NetworkGenerator(Protocol):
    """Structural interface for network generation.

    Implementations must be picklable (frozen dataclasses recommended)
    to support parallel execution via multiprocessing.
    """

    def generate(self, rng: np.random.Generator) -> csr_matrix:
        """Generate a single network.

        Args:
            rng: Random number generator for this realisation.

        Returns:
            Symmetric adjacency matrix in CSR format.
        """
        ...


@dataclass(frozen=True)
class ErdosRenyiGenerator:
    """Erdos-Renyi random graph G(n, p)."""

    n: int
    p: float

    def generate(self, rng: np.random.Generator) -> csr_matrix:
        from craeft.networks.generation.erdos_renyi import random_graph

        return random_graph(self.n, self.p, rng=rng)


@dataclass(frozen=True)
class ConfigurationModelGenerator:
    """Configuration model from a fixed degree sequence.

    When phi > 0, uses the Clustered Configuration Model to embed
    triangles and K4 cliques for the target clustering coefficient.
    """

    degrees: NDArray[np.int_]
    phi: float = 0.0

    def generate(self, rng: np.random.Generator) -> csr_matrix:
        from craeft.networks.generation.configuration_model import (
            configuration_model,
        )

        return configuration_model(self.degrees, phi=self.phi, rng=rng)


@dataclass(frozen=True)
class PoissonNetworkGenerator:
    """Configuration model with Poisson-sampled degrees.

    Stores only numeric parameters (no callables) so the generator
    is trivially picklable for multiprocessing.
    """

    n: int
    mean_degree: float
    max_degree: int
    phi: float = 0.0

    def generate(self, rng: np.random.Generator) -> csr_matrix:
        from scipy.stats import poisson

        from craeft.networks.generation.configuration_model import sample_network

        pmf = poisson(self.mean_degree).pmf
        return sample_network(self.n, pmf, self.max_degree, self.phi, rng)


@dataclass(frozen=True)
class BigVRewiringGenerator:
    """Compose a base generator with Big-V degree-preserving rewiring.

    Generates a network using `base`, then rewires edges to reach the
    target clustering coefficient while preserving the degree sequence.
    """

    base: NetworkGenerator
    target_clustering: float

    def generate(self, rng: np.random.Generator) -> csr_matrix:
        from craeft.networks.rewiring import big_v_rewire

        adjacency = self.base.generate(rng)
        return big_v_rewire(
            adjacency, target_clustering=self.target_clustering, rng=rng
        )


@dataclass(frozen=True)
class MotifDecompositionGenerator:
    """Generate networks via motif decomposition (tearing algorithm).

    Starts with disconnected cliques and rewires to reduce clustering
    from 1.0 to the target level.

    Args:
        num_nodes: Total nodes (must be divisible by clique_size).
        clique_size: Size of initial cliques.
        target_clustering: Desired global clustering coefficient.
    """

    num_nodes: int
    clique_size: int
    target_clustering: float

    def generate(self, rng: np.random.Generator) -> csr_matrix:
        from craeft.networks.rewiring import motif_decomposition

        return motif_decomposition(
            self.num_nodes, self.clique_size, self.target_clustering, rng=rng
        )


@dataclass(frozen=True)
class CMAGenerator:
    """Generate networks using the Cardinality Matching Algorithm.

    Accepts a degree sequence and subgraph specifications defining which
    motifs to embed and how many per node.

    Args:
        degrees: Per-node degree sequence.
        specs: Subgraph specs pairing motifs with participation sequences.
        connector: Connection strategy — "repeated", "refuse", or "erased".
    """

    degrees: NDArray[np.int_]
    specs: tuple  # tuple[SubgraphSpec, ...] — use tuple for picklability
    connector: str = "repeated"

    def generate(self, rng: np.random.Generator) -> csr_matrix:
        from craeft.networks.generation.configuration_model import cma

        return cma(
            self.degrees,
            list(self.specs),
            rng=rng,
            connector=self.connector,
        )
