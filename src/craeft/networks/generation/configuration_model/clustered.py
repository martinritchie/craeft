"""Clustered Configuration Model (CCM) for networks with prescribed clustering.

Implements the algorithm from Ritchie et al. (2014) "Higher-order structure
and epidemic dynamics in clustered networks", Journal of Theoretical Biology.

The CCM generates random graphs with a prescribed degree sequence AND a target
clustering coefficient by explicitly constructing triangle AND K4 (complete
square) motifs. This is crucial because K4 cliques contain 4 triangles each,
contributing significantly to clustering.
"""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.networks.generation.configuration_model.core import (
    edges_to_csr,
    sample_degree_sequence,
)


def _allocate_corners_degree5(
    n: int,
    phi: float,
    rng: np.random.Generator,
) -> tuple[NDArray[np.int_], NDArray[np.int_], NDArray[np.int_]]:
    """
    Allocate corners for homogeneous degree-5 networks per Ritchie et al. (2014).

    From Table 1 and Section 2.1.3 of the paper, for ⟨k⟩=5:
    - φ=0.2: p₁=0.5 (5 lines), p₃=0.5 (1 K4 corner + 1 triangle corner)
    - φ=0.4: p₃=1.0 (1 K4 corner + 1 triangle corner for all nodes)

    The relationship is: φ = 2p₃/5, so p₃ = 5φ/2.

    Args:
        n: Number of nodes.
        phi: Target clustering coefficient in [0, 0.4].
        rng: Random generator.

    Returns:
        Tuple of (single_stubs, triangle_corners, k4_corners) arrays per node.
    """
    # From paper: φ = 2p₃/5 for degree-5 networks
    # Therefore: p₃ = 5φ/2 (capped at 1.0)
    p3 = min(1.0, 5 * phi / 2)

    single_stubs = np.zeros(n, dtype=np.int_)
    triangle_corners = np.zeros(n, dtype=np.int_)
    k4_corners = np.zeros(n, dtype=np.int_)

    for i in range(n):
        if rng.random() < p3:
            # Configuration p₃: 1 K4 corner (3 stubs) + 1 triangle corner (2 stubs)
            k4_corners[i] = 1
            triangle_corners[i] = 1
        else:
            # Configuration p₁: 5 single stubs (all lines)
            single_stubs[i] = 5

    return single_stubs, triangle_corners, k4_corners


def _form_k4_cliques(
    k4_corners: NDArray[np.int_],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Form K4 cliques (complete squares) by matching nodes with K4 corners.

    A K4 clique has 4 nodes and 6 edges, containing 4 triangles.
    Each node contributes 3 stubs (degree 3 within the K4).

    Args:
        k4_corners: Number of K4 corners allocated to each node.
        rng: Random generator.

    Returns:
        Tuple of (rows, cols) arrays representing K4 edges.
    """
    # Create corner list: node i appears k4_corners[i] times
    corner_list = np.repeat(np.arange(len(k4_corners)), k4_corners)

    if len(corner_list) < 4:
        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

    # Shuffle and form K4s from consecutive quadruples
    rng.shuffle(corner_list)

    # Truncate to multiple of 4
    n_k4s = len(corner_list) // 4
    corner_list = corner_list[: n_k4s * 4]

    if n_k4s == 0:
        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

    # Reshape into K4s
    k4s = corner_list.reshape(n_k4s, 4)

    # Extract all 6 edges from each K4
    rows_list = []
    cols_list = []
    for k4 in k4s:
        n0, n1, n2, n3 = k4
        # All 6 edges of the complete graph K4
        rows_list.extend([n0, n0, n0, n1, n1, n2])
        cols_list.extend([n1, n2, n3, n2, n3, n3])

    return np.array(rows_list, dtype=np.int_), np.array(cols_list, dtype=np.int_)


def _form_triangles(
    corners: NDArray[np.int_],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Form triangles by matching nodes with available triangle corners.

    Args:
        corners: Number of triangle corners allocated to each node.
        rng: Random generator.

    Returns:
        Tuple of (rows, cols) arrays representing triangle edges.
        Each triangle (i, j, k) contributes edges (i,j), (j,k), (i,k).
    """
    # Create corner list: node i appears corners[i] times
    corner_list = np.repeat(np.arange(len(corners)), corners)

    if len(corner_list) < 3:
        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

    # Shuffle and form triangles from consecutive triples
    rng.shuffle(corner_list)

    # Truncate to multiple of 3
    n_triangles = len(corner_list) // 3
    corner_list = corner_list[: n_triangles * 3]

    if n_triangles == 0:
        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

    # Reshape into triangles
    triangles = corner_list.reshape(n_triangles, 3)

    # Extract edges from each triangle
    rows_list = []
    cols_list = []
    for tri in triangles:
        i, j, k = tri
        # Add all three edges of the triangle
        rows_list.extend([i, j, i])
        cols_list.extend([j, k, k])

    return np.array(rows_list, dtype=np.int_), np.array(cols_list, dtype=np.int_)


def _form_single_edges(
    stubs: NDArray[np.int_],
    rng: np.random.Generator,
) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
    """
    Form edges from single stubs using configuration model matching.

    Args:
        stubs: Number of single stubs per node.
        rng: Random generator.

    Returns:
        Tuple of (rows, cols) arrays representing single edges.
    """
    total_stubs = stubs.sum()
    if total_stubs < 2:
        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

    # Create stub list
    stub_list = np.repeat(np.arange(len(stubs)), stubs)

    # Ensure even number of stubs
    if len(stub_list) % 2 == 1:
        stub_list = stub_list[:-1]

    if len(stub_list) < 2:
        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

    # Shuffle and pair
    rng.shuffle(stub_list)
    rows = stub_list[0::2]
    cols = stub_list[1::2]

    return rows, cols


def clustered_configuration_model(
    degrees: NDArray[np.int_],
    phi: float = 0.0,
    rng: np.random.Generator | None = None,
    allow_self_loops: bool = False,
    allow_multi_edges: bool = False,
) -> csr_matrix:
    """
    Generate a random graph with prescribed degree sequence and clustering.

    Implements the Clustered Configuration Model (CCM) from Ritchie et al. (2014)
    which constructs networks by explicitly forming triangle AND K4 (complete
    square) motifs. K4 cliques are essential because each contains 4 triangles.

    For homogeneous degree-5 networks (the paper's focus):
    - φ=0.2: 50% of nodes get (1 K4 + 1 triangle), 50% get 5 single edges
    - φ=0.4: All nodes get (1 K4 + 1 triangle)

    Args:
        degrees: Degree sequence (length n array of non-negative integers).
            Sum must be even. Currently optimized for homogeneous degree-5.
        phi: Target clustering coefficient in [0, 1]. For degree-5 networks,
            maximum achievable is ~0.4 without overlapping motifs.
        rng: Optional random generator for reproducibility.
        allow_self_loops: If False, remove self-loops from the result.
        allow_multi_edges: If False, collapse multi-edges to single edges.

    Returns:
        Adjacency matrix in CSR format.

    Raises:
        ValueError: If degree sum is odd or phi is out of range.

    Example:
        >>> degrees = np.full(100, 5, dtype=np.int_)  # Homogeneous k=5
        >>> adj = clustered_configuration_model(
        ...     degrees, phi=0.4, rng=np.random.default_rng(42)
        ... )
    """
    rng = rng or np.random.default_rng()
    n = len(degrees)
    total_stubs = degrees.sum()

    if total_stubs % 2 != 0:
        msg = f"Degree sum must be even, got {total_stubs}"
        raise ValueError(msg)

    if not 0.0 <= phi <= 1.0:
        msg = f"phi must be in [0, 1], got {phi}"
        raise ValueError(msg)

    if total_stubs == 0:
        return csr_matrix((n, n), dtype=np.int8)

    # Check if this is a homogeneous degree-5 network
    is_homogeneous_k5 = np.all(degrees == 5)

    if is_homogeneous_k5:
        # Use paper's exact algorithm for degree-5 networks
        single_stubs, triangle_corners, k4_corners = _allocate_corners_degree5(
            n, phi, rng
        )

        # Form K4 cliques
        k4_rows, k4_cols = _form_k4_cliques(k4_corners, rng)

        # Form triangles
        tri_rows, tri_cols = _form_triangles(triangle_corners, rng)

        # Form single edges
        single_rows, single_cols = _form_single_edges(single_stubs, rng)

        # Combine all edges
        all_rows = np.concatenate([k4_rows, tri_rows, single_rows])
        all_cols = np.concatenate([k4_cols, tri_cols, single_cols])
    else:
        # Fallback: use simplified triangle-only approach for other degrees
        # This is a limitation - paper's algorithm is specific to degree-5
        triangle_corners = _allocate_triangle_corners_general(degrees, phi, rng)
        tri_rows, tri_cols = _form_triangles(triangle_corners, rng)

        # Remaining stubs
        remaining = degrees - 2 * triangle_corners
        remaining = np.maximum(remaining, 0)
        single_rows, single_cols = _form_single_edges(remaining, rng)

        all_rows = np.concatenate([tri_rows, single_rows])
        all_cols = np.concatenate([tri_cols, single_cols])

    return edges_to_csr(all_rows, all_cols, n, allow_self_loops, allow_multi_edges)


def _allocate_triangle_corners_general(
    degrees: NDArray[np.int_],
    phi: float,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """
    Fallback triangle allocation for non-degree-5 networks.

    This is a simplified approach that only forms triangles (no K4s).
    It will underestimate clustering for higher phi values.

    Args:
        degrees: Degree sequence.
        phi: Target clustering coefficient.
        rng: Random generator.

    Returns:
        Array of triangle corner counts per node.
    """
    n = len(degrees)
    max_corners = degrees // 2

    corners = np.zeros(n, dtype=np.int_)
    for i in range(n):
        if max_corners[i] > 0:
            corners[i] = rng.binomial(max_corners[i], phi)

    return corners


def sample_clustered_network(
    n: int,
    pmf: Callable[[NDArray[np.int_]], NDArray[np.floating]],
    max_degree: int,
    phi: float = 0.0,
    rng: np.random.Generator | None = None,
) -> csr_matrix:
    """
    Generate a clustered network with degrees sampled from a distribution.

    Convenience function that combines degree sampling with the CCM algorithm.

    Args:
        n: Number of nodes.
        pmf: Probability mass function P(D = k) for the degree distribution.
        max_degree: Maximum degree to consider.
        phi: Target clustering coefficient in [0, 1].
        rng: Optional random generator for reproducibility.

    Returns:
        Adjacency matrix in CSR format.

    Example:
        >>> from scipy.stats import poisson
        >>> adj = sample_clustered_network(
        ...     n=100,
        ...     pmf=poisson(4).pmf,
        ...     max_degree=20,
        ...     phi=0.3,
        ...     rng=np.random.default_rng(42)
        ... )
    """
    rng = rng or np.random.default_rng()
    degrees = sample_degree_sequence(n, pmf, max_degree, rng)
    return clustered_configuration_model(degrees, phi, rng)
