"""Configuration model for network generation.

Generates random graphs with prescribed degree sequences. When a clustering
coefficient (phi) is specified, the Clustered Configuration Model (CCM) is
used to embed subgraph structure (triangles, K4 cliques).

The main entry points are:

    configuration_model(degrees, phi=0.0, ...)
        Generate a graph from a degree sequence, optionally with clustering.

    sample_network(n, pmf, max_degree, phi=0.0, ...)
        Sample degrees from a distribution and generate a graph.

    sample_degree_sequence(n, pmf, max_degree, ...)
        Sample a degree sequence with guaranteed even sum.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.networks.generation.configuration_model.assembly import (
    AssemblyResult,
    assemble,
)
from craeft.networks.generation.configuration_model.clustered import (
    clustered_configuration_model as _clustered,
)
from craeft.networks.generation.configuration_model.connectors import (
    Connector,
    ConnectorName,
    Erased,
    Refuse,
    Repeated,
    get_connector,
)
from craeft.networks.generation.configuration_model.core import (
    configuration_model as _simple,
)
from craeft.networks.generation.configuration_model.core import (
    edges_to_csr,
    sample_degree_sequence,
)
from craeft.networks.generation.configuration_model.pairing import (
    Pairer,
    PairerName,
    PairingResult,
    get_pairer,
    pair_singles,
)
from craeft.networks.generation.configuration_model.partition import (
    MixedCardinalityError,
    StubPartition,
    partition_stubs,
)


def configuration_model(
    degrees: NDArray[np.int_],
    phi: float = 0.0,
    rng: np.random.Generator | None = None,
    allow_self_loops: bool = False,
    allow_multi_edges: bool = False,
) -> csr_matrix:
    """Generate a random graph with a prescribed degree sequence.

    When phi is 0.0 (default), generates a standard configuration model graph
    using random stub-pairing. When phi > 0, uses the Clustered Configuration
    Model to embed triangles and K4 cliques for the target clustering.

    Args:
        degrees: Degree sequence (length n, non-negative integers, even sum).
        phi: Target clustering coefficient in [0, 1]. Default 0.0 gives no
            clustering (standard configuration model).
        rng: Random generator for reproducibility.
        allow_self_loops: If False (default), remove self-loops.
        allow_multi_edges: If False (default), collapse multi-edges.

    Returns:
        Symmetric adjacency matrix in CSR format.
    """
    if phi == 0.0:
        return _simple(degrees, rng, allow_self_loops, allow_multi_edges)
    return _clustered(degrees, phi, rng, allow_self_loops, allow_multi_edges)


def sample_network(
    n: int,
    pmf: Callable[[NDArray[np.int_]], NDArray[np.floating]],
    max_degree: int,
    phi: float = 0.0,
    rng: np.random.Generator | None = None,
) -> csr_matrix:
    """Generate a network with degrees sampled from a distribution.

    Convenience function combining degree sampling with graph generation.
    Supports optional clustering via the phi parameter.

    Args:
        n: Number of nodes.
        pmf: Probability mass function P(D = k) for degrees.
        max_degree: Maximum degree to consider (truncates distribution).
        phi: Target clustering coefficient. Default 0.0 (no clustering).
        rng: Random generator for reproducibility.

    Returns:
        Symmetric adjacency matrix in CSR format.
    """
    rng = rng or np.random.default_rng()
    degrees = sample_degree_sequence(n, pmf, max_degree, rng)
    return configuration_model(degrees, phi, rng)


__all__ = [
    # Main API
    "configuration_model",
    "edges_to_csr",
    "sample_network",
    "sample_degree_sequence",
    # Assembly
    "AssemblyResult",
    "assemble",
    # Connectors
    "Connector",
    "ConnectorName",
    "Erased",
    "Refuse",
    "Repeated",
    "get_connector",
    # Pairing
    "Pairer",
    "PairerName",
    "PairingResult",
    "get_pairer",
    "pair_singles",
    # Partition
    "MixedCardinalityError",
    "StubPartition",
    "partition_stubs",
]
