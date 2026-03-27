"""Configuration model for generating random graphs with prescribed degree sequence."""

from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix


def sample_degree_sequence(
    n: int,
    pmf: Callable[[NDArray[np.int_]], NDArray[np.floating]],
    max_degree: int,
    rng: np.random.Generator | None = None,
) -> NDArray[np.int_]:
    """
    Sample a degree sequence guaranteed to have even sum.

    Uses conditional resampling: samples all degrees independently, then if the
    sum is odd, resamples one randomly chosen node from the parity-restricted
    distribution. This is unbiased and always completes in a single pass.

    Args:
        n: Number of nodes.
        pmf: Probability mass function P(D = k), vectorized over k.
            Should accept an array of degree values and return probabilities.
        max_degree: Maximum degree to consider (truncates distribution).
        rng: Random generator for reproducibility.

    Returns:
        Array of n degrees summing to an even number.

    Example:
        >>> # Poisson(3) degree distribution
        >>> from scipy.stats import poisson
        >>> rng = np.random.default_rng(42)
        >>> degrees = sample_degree_sequence(
        ...     100, poisson(3).pmf, max_degree=20, rng=rng
        ... )
        >>> degrees.sum() % 2
        0
    """
    rng = rng or np.random.default_rng()

    k_values = np.arange(max_degree + 1)
    probs = pmf(k_values)
    probs = probs / probs.sum()

    degrees = rng.choice(k_values, size=n, p=probs)

    if degrees.sum() % 2 == 0:
        return degrees

    # Resample one node from opposite-parity distribution
    # To flip sum parity, new degree must have opposite parity of old degree
    i = rng.integers(n)
    target_parity = 1 - (degrees[i] % 2)
    mask = (k_values % 2) == target_parity
    conditional_probs = np.where(mask, probs, 0.0)
    conditional_probs = conditional_probs / conditional_probs.sum()

    degrees[i] = rng.choice(k_values, p=conditional_probs)

    return degrees


def edges_to_csr(
    rows: NDArray[np.int_],
    cols: NDArray[np.int_],
    num_nodes: int,
    allow_self_loops: bool = False,
    allow_multi_edges: bool = False,
) -> csr_matrix:
    """Build a symmetric CSR adjacency matrix directly from one-directional edges.

    Constructs CSR arrays via an argsort-based row ordering, bypassing COO
    intermediate allocation. Each edge (i, j) is stored in both directions.

    Args:
        rows: Source node for each edge.
        cols: Target node for each edge.
        num_nodes: Number of nodes in the graph.
        allow_self_loops: If False, filter self-loops before construction.
        allow_multi_edges: If False, collapse multi-edges to single edges.

    Returns:
        Symmetric adjacency matrix in CSR format.
    """
    if len(rows) == 0:
        return csr_matrix((num_nodes, num_nodes), dtype=np.int8)

    if not allow_self_loops:
        mask = rows != cols
        rows = rows[mask]
        cols = cols[mask]

    # Symmetrise: each (i, j) becomes (i, j) and (j, i)
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    del rows, cols

    # Build indptr from row counts (before sorting frees sym_rows)
    indptr = np.empty(num_nodes + 1, dtype=np.int64)
    indptr[0] = 0
    np.cumsum(np.bincount(sym_rows, minlength=num_nodes), out=indptr[1:])

    # Sort column indices by row for CSR layout
    order = np.argsort(sym_rows, kind="quicksort")
    del sym_rows
    indices = sym_cols[order]
    del sym_cols, order

    data = np.ones(len(indices), dtype=np.int8)
    adj = csr_matrix((data, indices, indptr), shape=(num_nodes, num_nodes))

    if not allow_multi_edges:
        adj.sum_duplicates()
        adj.data = np.clip(adj.data, 0, 1)
        adj.eliminate_zeros()

    return adj


def configuration_model(
    degrees: NDArray[np.int_],
    rng: np.random.Generator | None = None,
    allow_self_loops: bool = False,
    allow_multi_edges: bool = False,
) -> csr_matrix:
    """
    Generate a random graph with a prescribed degree sequence.

    Uses the configuration model: creates stubs for each node according to
    its degree, then randomly pairs stubs to form edges.

    Args:
        degrees: Degree sequence (length n array of non-negative integers).
            Sum must be even.
        rng: Optional random generator for reproducibility.
        allow_self_loops: If False, remove self-loops from the result.
        allow_multi_edges: If False, collapse multi-edges to single edges.

    Returns:
        Adjacency matrix in CSR format.

    Raises:
        ValueError: If degree sum is odd (impossible to pair all stubs).

    Example:
        >>> degrees = np.array([2, 2, 2, 2])  # 4-cycle possible
        >>> adj = configuration_model(degrees, rng=np.random.default_rng(42))
        >>> adj.sum(axis=1).A1  # Realized degrees
        array([2, 2, 2, 2])
    """
    rng = rng or np.random.default_rng()
    n = len(degrees)
    total_stubs = degrees.sum()

    if total_stubs % 2 != 0:
        msg = f"Degree sum must be even, got {total_stubs}"
        raise ValueError(msg)

    if total_stubs == 0:
        return csr_matrix((n, n), dtype=np.int8)

    # Create stub list: node i appears degrees[i] times
    stubs = np.repeat(np.arange(n), degrees)

    # Randomly shuffle and pair consecutive stubs
    rng.shuffle(stubs)
    rows = stubs[0::2].copy()
    cols = stubs[1::2].copy()
    del stubs

    return edges_to_csr(rows, cols, n, allow_self_loops, allow_multi_edges)
