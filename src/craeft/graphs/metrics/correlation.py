"""Degree-degree correlation and degree-conditioned diagnostics.

Assortativity is a known second-order confound for clustered configuration
models: clustering potential is bounded by the degree-degree correlation
(Serrano & Boguna), and pushing clustered subgraphs onto high-degree nodes
measurably raises assortativity as a side effect. These metrics let a caller
check that a matched pair of networks differs only in the intended higher-order
structure, not silently in mixing pattern too.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix, triu

from craeft.graphs.metrics.clustering import SparseMatrix, local_clustering


def degree_assortativity(adjacency: SparseMatrix) -> float:
    """Pearson correlation of degrees at either end of an edge.

    Newman's r. Positive: high-degree nodes attach to high-degree nodes
    (assortative mixing). Negative: high-degree nodes attach to low-degree
    nodes (disassortative mixing, e.g. hub-and-spoke networks).

    Computed on the undirected edge list (one row per edge). The correlation
    terms (`j*k`, `j+k`, `j^2+k^2`) are already symmetric in the two
    endpoints, so each edge is counted once regardless of orientation.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Pearson correlation coefficient in [-1, 1]. Returns `nan` for a
        graph with no edges, or a degree-regular graph (zero variance in
        edge-end degrees makes the ratio ill-defined) — this is common for
        the recommended C-model family, which is degree-regular within each
        parity class. Use `clustering_by_degree` instead in that case.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> star = csr_matrix([[0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]])
        >>> degree_assortativity(star) < 0
        True
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    degrees = np.asarray(a.sum(axis=1)).ravel()

    coo = triu(a, k=1).tocoo()
    if coo.nnz == 0:
        return float("nan")

    j = degrees[coo.row]
    k = degrees[coo.col]

    mean_jk = np.mean(j * k)
    mean_half_sum = np.mean(0.5 * (j + k))
    mean_half_sq = np.mean(0.5 * (j**2 + k**2))

    numerator = mean_jk - mean_half_sum**2
    denominator = mean_half_sq - mean_half_sum**2

    if abs(denominator) < 1e-10:
        return float("nan")

    return float(numerator / denominator)


def average_neighbour_degree(adjacency: SparseMatrix) -> NDArray[np.floating]:
    """Per-node mean neighbour degree.

    For node i, the mean degree of i's neighbours. Binning this by a node's
    own degree gives the knn(k) profile used to diagnose mixing patterns.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Array of length n. Isolated nodes (degree 0) get 0.0.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> ring = csr_matrix([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
        >>> average_neighbour_degree(ring)
        array([2., 2., 2., 2.])
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    degrees = np.asarray(a.sum(axis=1)).ravel()
    neighbour_degree_sum = np.asarray(a.dot(degrees)).ravel()

    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(degrees > 0, neighbour_degree_sum / degrees, 0.0)

    return result


def clustering_by_degree(adjacency: SparseMatrix) -> dict[int, float]:
    """Mean local clustering coefficient per degree class — the c(k) profile.

    Composes `local_clustering` with a groupby on node degree. This is the
    diagnostic that exposes "clustered subgraphs pushed onto hubs": a skewed
    c(k) (high clustering concentrated at high k) signals that placement, and
    remains informative even when `degree_assortativity` is numerically
    unstable (e.g. degree-regular families).

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Mapping from degree value (present in the graph) to mean local
        clustering coefficient among nodes of that degree.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> tri = csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> clustering_by_degree(tri)
        {2: 1.0}
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    degrees = np.asarray(a.sum(axis=1)).ravel().astype(np.int_)
    local = local_clustering(a)

    result: dict[int, float] = {}
    for k in np.unique(degrees):
        mask = degrees == k
        result[int(k)] = float(local[mask].mean())
    return result
