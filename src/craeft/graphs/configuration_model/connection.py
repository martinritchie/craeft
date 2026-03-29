"""Edge formation for the configuration model family.

Two connection modes:
    - connect_subgraphs: form subgraph edges from typed hyperstub bins,
      checking for node duplicates and existing edges (CMA step 3)
    - connect_singles: pair remaining single stubs into simple edges,
      used by both vanilla CM and as CMA step 4

Reference:
    Ritchie et al. (2016), Algorithm 1 (p.278), lines 16-27.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

if TYPE_CHECKING:
    from craeft.graphs.configuration_model.models import SubgraphSequence
    from craeft.graphs.configuration_model.sequence import Allocation


# ---------------------------------------------------------------------------
# Subgraph connection (CMA)
# ---------------------------------------------------------------------------


def connect_subgraphs(
    sequences: list[SubgraphSequence],
    allocation: Allocation,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> tuple[NDArray[np.int_], NDArray[np.int_], set[tuple[int, int]]]:
    """Form subgraph edges from allocated hyperstub bins.

    For each subgraph spec, populates typed bins from the allocation,
    shuffles, and pops groups of nodes to form subgraph instances.
    Rejects selections containing node duplicates or edges that
    already exist. Reshuffles on collision.

    Args:
        sequences: Subgraph sequences.
        allocation: Hyperstub allocation from greedy matching.
        rng: Random number generator.
        max_attempts: Maximum consecutive failures before raising.

    Returns:
        Tuple of (rows, cols, existing_edges) where rows and cols
        are edge arrays and existing_edges is the set of formed
        edges in canonical (min, max) form.

    Raises:
        ConnectionError: If max_attempts exceeded for any subgraph.
    """
    ...


class ConnectionError(Exception):
    """Raised when the connection process cannot form valid subgraph instances."""


# ---------------------------------------------------------------------------
# Single stub pairing (vanilla CM + CMA)
# ---------------------------------------------------------------------------


def connect_singles(
    degrees: NDArray[np.int_],
    rng: np.random.Generator,
    existing_edges: set[tuple[int, int]] | None = None,
) -> csr_matrix:
    """Pair single stubs to form simple edges.

    Creates stubs for each node according to its degree, randomly
    shuffles them, then pairs consecutive stubs to form edges.
    Self-loops, multi-edges, and edges already in existing_edges
    are removed.

    For vanilla CM, call with the full degree sequence and no
    existing_edges. For CMA step 4, call with the remaining
    singles and the edges formed by connect_subgraphs.

    Args:
        degrees: Per-node stub counts. Sum must be even.
        rng: Random number generator.
        existing_edges: Edges to avoid (from prior subgraph
            connection). None for vanilla CM.

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

    # Remove edges that conflict with existing subgraph edges
    if existing_edges:
        keep = np.array(
            [
                (min(int(r), int(c)), max(int(r), int(c))) not in existing_edges
                for r, c in zip(rows, cols)
            ]
        )
        rows = rows[keep]
        cols = cols[keep]

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
