"""A priori clustering coefficient prediction from subgraph specifications.

Estimates the expected global clustering coefficient from the degree
sequence and subgraph specs without generating any networks.

Reference:
    Ritchie et al. (2015) "Generation and analysis of networks with a
    prescribed degree sequence and subgraph family", Section 2.1.1.
"""

import numpy as np
from numpy.typing import NDArray

from .subgraph_spec import SubgraphSpec


def predict_clustering(
    degrees: NDArray[np.int_],
    specs: list[SubgraphSpec],
) -> float:
    """Predict the expected global clustering coefficient.

    Uses the subgraph structure to count expected triangles and the
    degree distribution to count expected connected triples.

    Args:
        degrees: Per-node degree sequence.
        specs: Subgraph specifications.

    Returns:
        Expected global clustering coefficient in [0, 1].
    """
    total_triples = _count_triples(degrees)
    if total_triples == 0:
        return 0.0

    total_triangles = sum(_expected_triangles(spec) for spec in specs)

    return min(1.0, 3.0 * total_triangles / total_triples)


def _count_triples(degrees: NDArray[np.int_]) -> float:
    """Total expected connected triples from degree distribution."""
    return float(np.sum(degrees * (degrees - 1)))


def _triangles_in_motif(spec: SubgraphSpec) -> int:
    """Count triangles in the motif's adjacency matrix via trace(A^3)/6."""
    adj = spec.motif.adjacency.astype(np.float64)
    a2 = adj @ adj
    a3_trace = int(np.sum(a2 * adj))
    return a3_trace // 6


def _expected_triangles(spec: SubgraphSpec) -> float:
    """Expected total triangles contributed by a subgraph spec.

    Total instances = sum(sequence) / num_nodes (each instance uses
    num_nodes node-slots). Each instance contributes a fixed number
    of triangles determined by the motif's structure.
    """
    triangles_per_instance = _triangles_in_motif(spec)
    if triangles_per_instance == 0:
        return 0.0
    num_instances = spec.total / spec.motif.num_nodes
    return num_instances * triangles_per_instance
