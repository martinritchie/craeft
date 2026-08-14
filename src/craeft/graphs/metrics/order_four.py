"""Order-four (4-node) connected induced subgraph structure metrics.

Clustering (third order) is the quantity a designed pair of networks is
*matched* on. What actually distinguishes networks with identical degree
distribution and identical clustering is their order-four composition -- the
central result of Ritchie, Berthouze & Kiss (2014), *J. Theor. Biol.*.
Without these metrics, two networks that differ only in higher-order
structure look identical.

## Definitions (2014 Section 2.2, item 4)

Ratios of uniquely counted closed 4-node structures to all unique connected
4-node structures (open and closed):

- `phi_4_1` -- empty square (chordless 4-cycle)
- `phi_4_3` -- square with one diagonal (diamond)
- `phi_4_4` -- complete square (K4)
- `unclosed` (1 - phi_4) -- everything else

The paper counts each structure **uniquely** (once), not multiplicatively,
for tractability. See `order_four_ratios` for the caveat this creates.

## A sixth class not named by the paper's typology

Connected 4-node graphs actually come in six isomorphism classes, not five:
alongside path, star (both open/tree), and cycle/diamond/complete (the three
"square plus 0/1/2 diagonals" closed classes named above), there is **paw**
-- a triangle with a pendant edge. It cannot arise from this library's own
Hamiltonian-cycle-based subgraph patterns (see `graphs.base.Subgraph`), but
can appear incidentally in a generated graph. `count_order_four` reports it
explicitly; `order_four_ratios` folds it into the `unclosed` bucket, since it
is not one of the three named closed classes but does contain a triangle.

## Unique vs multiplicative counts (2014 Appendix A.3, open conjecture)

Unique counts (used here, matching the papers) are not what a pairwise ODE
closure needs. The 2014 paper conjectures -- but does not establish -- that
the conversion factor to a multiplicative count is each structure's
automorphism group order: 6 (triangle), 2 (3-path), 8 (empty square), 4
(diamond), 24 (K4). This module does not implement that conversion; if a
caller needs it, it should be applied explicitly and the conjecture cited as
open, not treated as established.
"""

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.graphs.metrics.clustering import SparseMatrix

# Census index -> class name, pinned against the installed igraph by
# tests/graphs/metrics/test_order_four.py::TestCensusIndexMapping (built in
# isolation, only the expected key non-zero for each of the six classes).
# Index order is an igraph implementation detail, not part of its public
# contract -- do not assume this is stable across igraph versions without
# re-running that test.
_CENSUS_STAR = 4
_CENSUS_PATH = 6
_CENSUS_PAW = 7
_CENSUS_CYCLE = 8
_CENSUS_DIAMOND = 9
_CENSUS_COMPLETE = 10

# Degree sequence (sorted, within the 4-node induced subgraph) -> class name.
# Used by order_four_per_node, which enumerates subsets directly rather than
# going through igraph's motif census (no per-vertex breakdown is exposed by
# motifs_randesu).
_CLASS_BY_DEGREE_SEQUENCE = {
    (1, 1, 2, 2): "path",
    (1, 1, 1, 3): "star",
    (2, 2, 2, 2): "cycle",
    (1, 2, 2, 3): "paw",
    (2, 2, 3, 3): "diamond",
    (3, 3, 3, 3): "complete",
}


def _to_igraph(adjacency: SparseMatrix) -> ig.Graph:
    a = csr_matrix(adjacency, dtype=np.float64)
    coo = a.tocoo()
    edges = [(int(i), int(j)) for i, j in zip(coo.row, coo.col) if i < j]
    return ig.Graph(n=a.shape[0], edges=edges, directed=False)


def count_order_four(adjacency: SparseMatrix) -> dict[str, int]:
    """Unique counts of connected 4-node induced subgraphs, by isomorphism class.

    Uses igraph's motif census (`motifs_randesu(size=4)`), which is C-speed
    and enumerates each connected induced subgraph exactly once. Disconnected
    census classes (which igraph returns as `nan`) are dropped.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Dict with keys `path`, `star`, `cycle`, `paw`, `diamond`, `complete`
        (unique counts per class), plus `open_total` (path + star) and
        `closed_total` (cycle + diamond + complete). `paw` is reported but
        deliberately excluded from both totals -- see the module docstring.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> k4 = csr_matrix(np.ones((4, 4)) - np.eye(4))
        >>> count_order_four(k4)["complete"]
        1
        >>> count_order_four(k4)["diamond"]
        0
    """
    g = _to_igraph(adjacency)
    census = g.motifs_randesu(size=4)

    def _at(index: int) -> int:
        value = census[index]
        return 0 if (isinstance(value, float) and np.isnan(value)) else int(value)

    counts = {
        "path": _at(_CENSUS_PATH),
        "star": _at(_CENSUS_STAR),
        "cycle": _at(_CENSUS_CYCLE),
        "paw": _at(_CENSUS_PAW),
        "diamond": _at(_CENSUS_DIAMOND),
        "complete": _at(_CENSUS_COMPLETE),
    }
    counts["open_total"] = counts["path"] + counts["star"]
    counts["closed_total"] = counts["cycle"] + counts["diamond"] + counts["complete"]
    return counts


def order_four_ratios(adjacency: SparseMatrix) -> dict[str, float]:
    """phi_4^1, phi_4^3, phi_4^4 and 1 - phi_4 (2014 Section 2.2, item 4).

    Denominator is the *true* total of connected 4-node induced subgraphs
    (all six classes, including `paw` -- see the module docstring), so the
    four returned values always sum to 1.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Dict with keys `phi_4_1` (empty square / chordless 4-cycle),
        `phi_4_3` (diamond), `phi_4_4` (complete / K4), and `unclosed`
        (1 - phi_4: path, star and paw combined). All zero if the network
        has no connected 4-node induced subgraphs.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> k4 = csr_matrix(np.ones((4, 4)) - np.eye(4))
        >>> order_four_ratios(k4)["phi_4_4"]
        1.0
    """
    counts = count_order_four(adjacency)
    total = counts["open_total"] + counts["closed_total"] + counts["paw"]

    if total == 0:
        return {"phi_4_1": 0.0, "phi_4_3": 0.0, "phi_4_4": 0.0, "unclosed": 0.0}

    return {
        "phi_4_1": counts["cycle"] / total,
        "phi_4_3": counts["diamond"] / total,
        "phi_4_4": counts["complete"] / total,
        "unclosed": (total - counts["closed_total"]) / total,
    }


def _connected_4subsets(adjacency_lists: list[set[int]], n: int) -> set[frozenset[int]]:
    """Every connected 4-node vertex subset of the graph, each exactly once."""
    results: set[frozenset[int]] = set()

    def grow(subset: frozenset[int]) -> None:
        if len(subset) == 4:
            results.add(subset)
            return
        frontier: set[int] = set()
        for node in subset:
            frontier |= adjacency_lists[node]
        frontier -= subset
        for candidate in frontier:
            grow(subset | {candidate})

    for start in range(n):
        grow(frozenset({start}))
    return results


def order_four_per_node(adjacency: SparseMatrix) -> dict[str, NDArray[np.int_]]:
    """Per-node unique counts of connected 4-node induced subgraphs, by class.

    For each connected 4-node vertex subset, classifies it by its sorted
    induced-degree sequence and increments the count for every member node.
    The 2014 Fig. 8 per-node distributions.

    This enumerates connected subsets by growing them one adjacent node at a
    time, so cost scales with the number of such subsets in the network
    (roughly `n * mean_degree^3` for sparse graphs) -- intended for
    diagnostic use on moderate-size networks, not as a replacement for the
    global igraph-backed `count_order_four` on large or dense graphs.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Dict with keys `path`, `star`, `cycle`, `paw`, `diamond`, `complete`,
        each an array of length n giving each node's participation count.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> square = csr_matrix([[0,1,0,1],[1,0,1,0],[0,1,0,1],[1,0,1,0]])
        >>> order_four_per_node(square)["cycle"]
        array([1, 1, 1, 1])
    """
    a = csr_matrix(adjacency, dtype=np.float64)
    n = a.shape[0]
    adjacency_lists = [
        set(a.indices[a.indptr[i] : a.indptr[i + 1]].tolist()) for i in range(n)
    ]

    classes = ("path", "star", "cycle", "paw", "diamond", "complete")
    result: dict[str, NDArray[np.int_]] = {
        cls: np.zeros(n, dtype=np.int_) for cls in classes
    }

    for subset in _connected_4subsets(adjacency_lists, n):
        nodes = tuple(subset)
        degrees = tuple(
            sorted(
                sum(
                    1
                    for other in nodes
                    if other != node and other in adjacency_lists[node]
                )
                for node in nodes
            )
        )
        cls = _CLASS_BY_DEGREE_SEQUENCE.get(degrees)
        if cls is None:
            continue
        for node in nodes:
            result[cls][node] += 1

    return result
