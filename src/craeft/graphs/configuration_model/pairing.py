"""Stub pairing algorithm for the configuration model."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix


def pair_stubs(
    degrees: NDArray[np.int_],
    rng: np.random.Generator,
) -> csr_matrix:
    """Generate a random graph by pairing stubs.

    Creates stubs for each node according to its degree, randomly
    shuffles them, then pairs consecutive stubs to form edges.
    Self-loops and multi-edges are removed.

    Args:
        degrees: Per-node degree sequence. Sum must be even.
        rng: Random number generator.

    Returns:
        Symmetric adjacency matrix in CSR format.

    Raises:
        ValueError: If degree sum is odd.
    """
    n = len(degrees)
    total_stubs = int(degrees.sum())

    if total_stubs % 2 != 0:
        msg = f"Degree sum must be even, got {total_stubs}"
        raise ValueError(msg)

    if total_stubs == 0:
        return csr_matrix((n, n), dtype=np.int8)

    stubs = np.repeat(np.arange(n), degrees)
    rng.shuffle(stubs)

    rows = stubs[0::2]
    cols = stubs[1::2]

    # Remove self-loops
    mask = rows != cols
    rows = rows[mask]
    cols = cols[mask]

    if len(rows) == 0:
        return csr_matrix((n, n), dtype=np.int8)

    # Build symmetric adjacency
    sym_rows = np.concatenate([rows, cols])
    sym_cols = np.concatenate([cols, rows])
    data = np.ones(len(sym_rows), dtype=np.int8)

    adj = coo_matrix((data, (sym_rows, sym_cols)), shape=(n, n), dtype=np.int8).tocsr()

    # Collapse multi-edges
    adj.data = np.clip(adj.data, 0, 1)
    adj.eliminate_zeros()

    return adj
