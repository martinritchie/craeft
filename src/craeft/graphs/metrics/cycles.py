"""Induced (chordless) cycle counting.

A cycle is *induced* when the only edges among its nodes are the cycle edges
themselves — no chords. That is the right notion for this library: a designed
C5 that happens to acquire a chord is no longer a C5, and counting it as one
would credit the generator with structure it did not produce. The chorded
variants are order-four (and higher) classes in their own right, reported by
`craeft.graphs.metrics.order_four`.

At length 3 every cycle is trivially chordless, so `induced_cycle_count(a, 3)`
is the ordinary triangle count and `cycles_per_node(a, 3)` agrees elementwise
with `clustering.triangles_per_node`.

Both functions enumerate cycles the same way, so the identity

    cycles_per_node(a, L).sum() == L * induced_cycle_count(a, L)

holds exactly. Enumeration cost scales roughly as `n * mean_degree**(L - 1)`;
these are diagnostic metrics for moderate-size networks, not large ones.
"""

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.graphs.metrics.clustering import SparseMatrix

MIN_CYCLE_LENGTH = 3


def _to_igraph(adjacency: SparseMatrix) -> ig.Graph:
    """Simple undirected igraph view of a sparse adjacency matrix."""
    a = csr_matrix(adjacency, dtype=np.float64)
    coo = a.tocoo()
    edges = [(int(i), int(j)) for i, j in zip(coo.row, coo.col) if i < j]
    g = ig.Graph(n=a.shape[0], edges=edges, directed=False)
    g.simplify()
    return g


def _enumerate_induced_cycles(
    adjacency: SparseMatrix, length: int
) -> list[tuple[int, ...]]:
    """Every induced cycle of `length`, each yielded exactly once.

    Depth-first path extension rooted at the cycle's lowest-numbered node.
    Each cycle is reachable by two traversals (one per direction); the
    `path[1] < path[-1]` guard keeps one of them.
    """
    if length < MIN_CYCLE_LENGTH:
        raise ValueError(f"Cycle length must be at least {MIN_CYCLE_LENGTH}")

    g = _to_igraph(adjacency)
    neighbours = [set(g.neighbors(v)) for v in range(g.vcount())]

    def _is_chordless(path: list[int]) -> bool:
        for i in range(length):
            for j in range(i + 2, length):
                if i == 0 and j == length - 1:
                    continue  # the closing edge, not a chord
                if path[j] in neighbours[path[i]]:
                    return False
        return True

    cycles: list[tuple[int, ...]] = []
    for start in range(g.vcount()):
        stack: list[tuple[int, list[int]]] = [(start, [start])]
        while stack:
            node, path = stack.pop()
            if len(path) == length:
                if (
                    start in neighbours[node]
                    and path[1] < path[-1]
                    and _is_chordless(path)
                ):
                    cycles.append(tuple(path))
                continue
            for candidate in neighbours[node]:
                if candidate <= start or candidate in path:
                    continue
                stack.append((candidate, [*path, candidate]))
    return cycles


def induced_cycle_count(adjacency: SparseMatrix, length: int) -> int:
    """Number of unique induced (chordless) cycles of `length`.

    Each cycle is counted once, irrespective of starting node or traversal
    direction. At `length=3` this is the ordinary triangle count; above it,
    chorded cycles are excluded — a diamond (K4 minus an edge) contains no
    induced 4-cycle.

    Args:
        adjacency: Symmetric adjacency matrix. Weights, self-loops and
            repeated edges are ignored; the graph is treated as simple.
        length: Cycle length, at least 3.

    Returns:
        Count of distinct induced cycles of the requested length.

    Raises:
        ValueError: If `length` is less than 3.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> square = csr_matrix([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
        >>> induced_cycle_count(square, 4)
        1
        >>> diamond = csr_matrix([[0,1,1,0],[1,0,1,1],[1,1,0,1],[0,1,1,0]])
        >>> induced_cycle_count(diamond, 4)
        0
    """
    return len(_enumerate_induced_cycles(adjacency, length))


def cycles_per_node(adjacency: SparseMatrix, length: int) -> NDArray[np.int_]:
    """Number of unique induced cycles of `length` through each node.

    Generalises `triangles_per_node`, with which it agrees elementwise at
    `length=3`. Every induced cycle is enumerated once and increments each of
    its `length` member nodes, so

        cycles_per_node(a, L).sum() == L * induced_cycle_count(a, L)

    holds exactly.

    Args:
        adjacency: Symmetric adjacency matrix.
        length: Cycle length, at least 3.

    Returns:
        Array of length n giving each node's induced-cycle participation.

    Raises:
        ValueError: If `length` is less than 3.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> square = csr_matrix([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
        >>> cycles_per_node(square, 4)
        array([1, 1, 1, 1])
    """
    n = csr_matrix(adjacency).shape[0]
    counts = np.zeros(n, dtype=np.int_)
    for cycle in _enumerate_induced_cycles(adjacency, length):
        for node in cycle:
            counts[node] += 1
    return counts
