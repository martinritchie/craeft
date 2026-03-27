"""Clustering coefficient and triangle counting for networks."""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_array, csr_matrix, spmatrix

# Type alias for sparse adjacency matrices
SparseMatrix = spmatrix | csr_array


def count_triangles(adjacency: SparseMatrix) -> int:
    """
    Count the total number of triangles in an undirected network.

    Uses the trace formula: triangles = trace(A³) / 6, where A is the
    adjacency matrix. Each triangle is counted 6 times (once per vertex
    and once per direction).

    Args:
        adjacency: Symmetric adjacency matrix (sparse or dense).

    Returns:
        Number of triangles in the network.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> # Triangle: 0-1-2-0
        >>> adj = csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> count_triangles(adj)
        1
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    a2 = a @ a
    a3 = a2 @ a
    return int(a3.trace()) // 6


def triangles_per_node(adjacency: SparseMatrix) -> NDArray[np.int_]:
    """
    Count triangles incident to each node.

    For node i, counts the number of triangles where i is a vertex.
    This equals (A³)_{ii} / 2 since each triangle at i is traversed
    in both directions.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Array of length n with triangle counts per node.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> # Triangle: 0-1-2-0, plus edge 0-3
        >>> adj = csr_matrix([
        ...     [0, 1, 1, 1],
        ...     [1, 0, 1, 0],
        ...     [1, 1, 0, 0],
        ...     [1, 0, 0, 0]
        ... ])
        >>> triangles_per_node(adj)
        array([1, 1, 1, 0])
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    a2 = a @ a
    a3_diag = np.asarray(a2.multiply(a).sum(axis=1)).ravel()
    return (a3_diag // 2).astype(np.int_)


def local_clustering(adjacency: SparseMatrix) -> NDArray[np.floating]:
    """
    Compute local clustering coefficient for each node.

    The local clustering coefficient C_i measures how close node i's
    neighbours are to forming a clique:

        C_i = 2 * T_i / (k_i * (k_i - 1))

    where T_i is the number of triangles containing i, and k_i is i's
    degree. Nodes with degree < 2 have clustering coefficient 0.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Array of length n with clustering coefficients in [0, 1].

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> # Complete graph K3 (triangle)
        >>> adj = csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
        >>> local_clustering(adj)
        array([1., 1., 1.])
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    degrees = np.asarray(a.sum(axis=1)).ravel()
    tri = triangles_per_node(a)

    # Maximum possible triangles for each node
    max_triangles = degrees * (degrees - 1) / 2

    # Avoid division by zero for degree < 2
    with np.errstate(divide="ignore", invalid="ignore"):
        clustering = np.where(max_triangles > 0, tri / max_triangles, 0.0)

    return clustering


def global_clustering_coefficient(adjacency: SparseMatrix) -> float:
    """
    Compute the global clustering coefficient (transitivity).

    Defined as the ratio of closed triplets (triangles × 3) to all
    connected triplets:

        C = 3 * (number of triangles) / (number of connected triples)

    This measures the overall tendency for nodes to cluster together.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Global clustering coefficient in [0, 1].

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> # Complete graph K4
        >>> adj = csr_matrix([
        ...     [0, 1, 1, 1],
        ...     [1, 0, 1, 1],
        ...     [1, 1, 0, 1],
        ...     [1, 1, 1, 0]
        ... ])
        >>> global_clustering_coefficient(adj)
        1.0
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    degrees = np.asarray(a.sum(axis=1)).ravel()

    # Number of connected triples centered at each node: k*(k-1)/2
    triples = (degrees * (degrees - 1) / 2).sum()

    if triples == 0:
        return 0.0

    triangles = count_triangles(a)
    return float(3 * triangles / triples)
