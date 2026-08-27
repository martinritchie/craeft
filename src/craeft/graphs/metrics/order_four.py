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

- `phi_4_1` -- **the aggregate**: proportion of *all* closed quadruples
- `phi_4_2` -- empty square (chordless 4-cycle)
- `phi_4_3` -- square with one diagonal (diamond)
- `phi_4_4` -- complete square (K4)
- `unclosed` (1 - phi_4_1) -- everything else

with `phi_4_1 == phi_4_2 + phi_4_3 + phi_4_4`, an identity 2014 Table 2
satisfies row by row.

.. warning::
   Until ticket 008 this module returned the **empty square** under the key
   `phi_4_1`, and returned no aggregate at all. That numbering came from a
   ``pdftotext`` transcription that dropped a superscript; the published PDF
   (p. 24 Section 2.2 item 4, and Table 2 on p. 28) is authoritative and is
   what the keys above now follow. Any recorded `phi_4_1` value from before
   that fix is an empty-square ratio and should be read as `phi_4_2`.

The paper counts each structure **uniquely** (once), not multiplicatively,
for tractability. See below for the caveat this creates.

## The denominator, and why these values are not comparable to 2014 Table 2

The paper gives two different denominators one sentence apart (p. 24, item 4):
first *"to all connected structures of 4 nodes"*, then *"global ratios of
unique order-four structure counts to all unique paths counts, closed and
unclosed"*. These are **not the same set**. Appendix A.2's algorithm is path
extension, so it implements the second reading -- and a star K(1,3) contains
no 4-node path, so it can never be enumerated by it. (Appendix A.3's eq. (23)
confirms this from the paper's own algebra: a star scores exactly 0 there.)

This module implements the **first** reading: the denominator is the count of
connected 4-node *induced subgraphs*, all six isomorphism classes, stars
included. Measured on this library's own generated families, **stars are
21-27% of that denominator**, so the two readings differ by roughly 30-36% in
every phi_4^i. A second-order difference compounds it: the paper counts
*paths*, and one induced subgraph contains several distinct ones (a paw
contains 2, a diamond 12), so its denominator is not a subgraph count at all.

**Consequence: craeft's phi_4^i are internally consistent and comparable
across generated families, but are NOT numerically comparable with 2014
Table 2.** A paper-faithful path-count variant is deliberately not
implemented -- it needs A.2's path enumeration with circular- and
reverse-permutation elimination, not a subgraph census. Raise a ticket if
reproducing Table 2's absolute values ever becomes a requirement.

## A sixth class not named by the paper's typology

Connected 4-node graphs come in six isomorphism classes, not five: alongside
path, star (both open/tree), and cycle/diamond/complete (the three "square
plus 0/1/2 diagonals" closed classes named above), there is **paw** -- a
triangle with a pendant edge, catalogued as **G6** in `docs/concepts/motifs.md`
(Przulj graphlet notation).

Both `count_order_four` and `order_four_ratios` report it explicitly, and
`order_four_ratios` *also* counts it inside `unclosed`. That is not a choice
this library makes: the paper defines its ratios over *"4-node structures
connected in a loop"*, and a paw contains no 4-cycle, so it is unclosed by
the paper's own definition. `paw` is therefore a **component of `unclosed`**,
not a seventh term -- the five terms `phi_4_2 + phi_4_3 + phi_4_4 + unclosed`
already sum to 1 without it.

It is worth reporting because it is the most sensitive order-four
discriminator available for triangle-bearing families: a paw is the
by-product signature of a designed triangle (any triangle plus one external
edge makes one). Measured at n=1000, <k>=4 (`dev/control_audit.py`
section G): 0.59-0.68% of connected quadruples in the {Null, C4, C5, C6}
cycle families and flat across all four, but **7.4% (diamond) and 15.3%
(K4)**, where paws outnumber the *designed* structures 21x and 40x
respectively.

A paw cannot arise from this library's own subgraph patterns -- the CCM
requires a Hamiltonian cycle and G6 has none -- so it is structurally
impossible as an *input* subgraph, and every paw observed is a by-product.

## Unique vs multiplicative counts (2014 Appendix A.3, open conjecture)

Unique counts (used here, matching the papers) are not what a pairwise ODE
closure needs. The 2014 paper conjectures -- but does not establish -- that
the conversion factor to a multiplicative count is each structure's
automorphism group order: 6 (triangle), 2 (3-path), 8 (empty square), 4
(diamond), 24 (K4). The two 4-node classes the paper never names have, by
this project's derivation, |Aut| = 2 (paw) and 6 (star).

.. warning::
   **The conversion factor is identity-dependent.** Those 4/8/24 figures
   apply to the *individual* identities, eqs. (24)-(26). Under the
   *aggregate* identity eq. (23) the weights differ: measured directly, it
   counts a 3-path 2x, an empty square 8x, a **diamond 12x** (not 4), a
   **paw 4x** (not 2), a K4 24x and a **star 0x**. So "paw = 2" is right as
   an automorphism order and **wrong** as an eq.-(23) conversion factor.

   Eq. (23)'s left-hand side enumerates exactly four classes and **omits the
   paw**, which its right-hand side nonetheless counts 4x each. That is a
   defect in the published paper, not a transcription artifact; anyone using
   eq. (23) as the multiplicative denominator undercounts by the paw term.

This module implements no such conversion; if a caller needs one, it should
be applied explicitly, against the right identity, and the conjecture cited
as open rather than established.
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
    """phi_4^1..phi_4^4, 1 - phi_4^1 and paw (2014 Section 2.2, item 4).

    Denominator is the *true* total of connected 4-node induced subgraphs
    (all six classes, `paw` and `star` included). **This is one of the
    paper's two mutually inconsistent denominators, and the values are
    therefore not comparable with 2014 Table 2** -- see the module docstring,
    which quotes both phrasings and gives the measured size of the gap.

    Args:
        adjacency: Symmetric adjacency matrix.

    Returns:
        Dict with keys `phi_4_1` (aggregate: all closed quadruples),
        `phi_4_2` (empty square / chordless 4-cycle), `phi_4_3` (diamond),
        `phi_4_4` (complete / K4), `unclosed` (1 - phi_4_1: path, star and
        paw combined) and `paw` (diagnostic; a *component* of `unclosed`,
        not an additional term). All zero if the network has no connected
        4-node induced subgraphs.

        Invariants: `phi_4_2 + phi_4_3 + phi_4_4 == phi_4_1`,
        `phi_4_1 + unclosed == 1`, and `paw <= unclosed`.

    Example:
        >>> from scipy.sparse import csr_matrix
        >>> k4 = csr_matrix(np.ones((4, 4)) - np.eye(4))
        >>> order_four_ratios(k4)["phi_4_4"]
        1.0
        >>> order_four_ratios(k4)["phi_4_1"]
        1.0
    """
    counts = count_order_four(adjacency)
    total = counts["open_total"] + counts["closed_total"] + counts["paw"]

    keys = ("phi_4_1", "phi_4_2", "phi_4_3", "phi_4_4", "unclosed", "paw")
    if total == 0:
        return dict.fromkeys(keys, 0.0)

    return {
        "phi_4_1": counts["closed_total"] / total,
        "phi_4_2": counts["cycle"] / total,
        "phi_4_3": counts["diamond"] / total,
        "phi_4_4": counts["complete"] / total,
        "unclosed": (total - counts["closed_total"]) / total,
        "paw": counts["paw"] / total,
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
