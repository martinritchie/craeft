"""Tests for order-four (4-node) connected induced subgraph structure metrics."""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from craeft.graphs.base import Subgraph
from craeft.graphs.metrics import (
    count_order_four,
    order_four_per_node,
    order_four_ratios,
    unique_triangles,
)

# ---------------------------------------------------------------------------
# Helpers: isolated single-instance graphs of each connected 4-node class
# ---------------------------------------------------------------------------


def _from_edges(n: int, edges: list[tuple[int, int]]) -> csr_matrix:
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _path4() -> csr_matrix:
    return _from_edges(4, [(0, 1), (1, 2), (2, 3)])


def _star4() -> csr_matrix:
    return _from_edges(4, [(0, 1), (0, 2), (0, 3)])


def _cycle4() -> csr_matrix:
    return _from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)])


def _diamond4() -> csr_matrix:
    return _from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])


def _complete4() -> csr_matrix:
    return _from_edges(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])


def _paw4() -> csr_matrix:
    """Triangle {0,1,2} plus a pendant edge to node 3.

    Not named in the ticket's census table, but a real 6th connected
    4-node isomorphism class distinct from the other five (confirmed
    against the installed igraph: census index 7, disjoint from star=4,
    path=6, cycle=8, diamond=9, complete=10). It cannot arise from the
    library's own Hamiltonian-cycle-based subgraph patterns, but can arise
    incidentally in a generated graph (e.g. a triangle subgraph plus one
    extra edge from random stub pairing that doesn't complete a diamond).
    """
    return _from_edges(4, [(0, 1), (1, 2), (2, 0), (2, 3)])


def _cycle_n(n: int) -> csr_matrix:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return _from_edges(n, edges)


def _disjoint_cycles(k_copies: int, cycle_len: int) -> csr_matrix:
    """k disjoint copies of a cycle_len-cycle."""
    n = k_copies * cycle_len
    edges = []
    for c in range(k_copies):
        base = c * cycle_len
        for i in range(cycle_len):
            edges.append((base + i, base + (i + 1) % cycle_len))
    return _from_edges(n, edges)


CLASS_BUILDERS = {
    "path": _path4,
    "star": _star4,
    "cycle": _cycle4,
    "paw": _paw4,
    "diamond": _diamond4,
    "complete": _complete4,
}


# ---------------------------------------------------------------------------
# 1. Census index mapping is pinned by construction, not assumed
# ---------------------------------------------------------------------------


class TestCensusIndexMapping:
    """Each isolated pattern must classify as exactly one class.

    This is the load-bearing test: it exercises the module's internal
    census-index -> class-name mapping against the installed igraph. If the
    mapping constants were wrong, one of these would show the wrong key
    non-zero (or all-zero) instead of exactly the expected one.
    """

    @pytest.mark.parametrize(("expected_class", "builder"), CLASS_BUILDERS.items())
    def test_only_expected_class_is_nonzero(self, expected_class, builder) -> None:
        counts = count_order_four(builder())
        for cls in CLASS_BUILDERS:
            expected_count = 1 if cls == expected_class else 0
            assert counts[cls] == expected_count, (
                f"{expected_class} pattern: expected {cls}={expected_count}, "
                f"got {counts[cls]}"
            )


# ---------------------------------------------------------------------------
# 2. Diamond has two triangles (consistency with ticket 004's unique_triangles)
# ---------------------------------------------------------------------------


class TestDiamondContainsTwoTriangles:
    def test_consistency_with_unique_triangles(self) -> None:
        adj = _diamond4()
        assert count_order_four(adj)["diamond"] == 1

        subgraph = Subgraph(
            adjacency=np.array(
                [[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]]
            )
        )
        assert unique_triangles(subgraph) == 2


# ---------------------------------------------------------------------------
# 3. K4 counts as complete only -- induced semantics, not "contains a diamond"
# ---------------------------------------------------------------------------


class TestInducedSemantics:
    def test_k4_is_complete_not_diamond(self) -> None:
        counts = count_order_four(_complete4())
        assert counts["complete"] == 1
        assert counts["diamond"] == 0
        assert counts["cycle"] == 0
        assert counts["paw"] == 0

    def test_k4_closed_total_is_one(self) -> None:
        counts = count_order_four(_complete4())
        assert counts["closed_total"] == 1
        assert counts["open_total"] == 0


# ---------------------------------------------------------------------------
# 4. Ratios sum to one
# ---------------------------------------------------------------------------


class TestRatiosSumToOne:
    """phi_4^1 + phi_4^3 + phi_4^4 + (1 - phi_4) == 1.

    Two judgment calls documented here, both against the ticket's own text:

    1. The ticket's TDD list also mentions a "phi_4^2" term, but its own
       definitions table only names phi_4^1 (empty square), phi_4^3
       (diamond), phi_4^4 (complete), and 1 - phi_4 (unclosed). There are
       exactly three isomorphism classes built from "a square plus 0/1/2
       diagonals" -- no room for a fourth -- so phi_4^2 looks like a typo
       carried over from the paper's original numbering. This test follows
       the definitions table, not the TDD list.

    2. The ticket's census-index table lists five classes (star, path,
       cycle, diamond, complete), but the installed igraph's motifs_randesu
       census actually has six non-degenerate classes: "paw" (a triangle
       with a pendant edge, census index 7) is real and distinct from
       diamond. It never arises from this library's own Hamiltonian-cycle
       subgraph patterns, but can appear incidentally in a generated graph.
       Since it's not one of the three named closed classes, it is folded
       into the "unclosed" denominator alongside path and star -- this is
       the only choice that keeps phi_4^1 + phi_4^3 + phi_4^4 + unclosed
       exactly 1 over the *true* population of connected quadruples (folding
       it out of the total instead would silently inflate the phi_4^i
       values whenever paws are present).
    """

    @pytest.mark.parametrize("builder", CLASS_BUILDERS.values())
    def test_single_instance(self, builder) -> None:
        ratios = order_four_ratios(builder())
        total = (
            ratios["phi_4_1"]
            + ratios["phi_4_3"]
            + ratios["phi_4_4"]
            + ratios["unclosed"]
        )
        assert total == pytest.approx(1.0)

    def test_mixed_population(self) -> None:
        adj = _disjoint_cycles(3, 4)
        ratios = order_four_ratios(adj)
        total = (
            ratios["phi_4_1"]
            + ratios["phi_4_3"]
            + ratios["phi_4_4"]
            + ratios["unclosed"]
        )
        assert total == pytest.approx(1.0)

    def test_empty_graph_is_zero_not_nan(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.float64)
        ratios = order_four_ratios(adj)
        assert all(v == 0.0 for v in ratios.values())


# ---------------------------------------------------------------------------
# 5. The recommended {Null, C4, C5, C6} family: phi_4^1 separates C4 (the
#    ticket's real purpose -- distinguishing networks matched on degree and
#    clustering by their order-four composition)
# ---------------------------------------------------------------------------


class TestEmptyCycleFamilyDiffersInOrderFour:
    def test_c4_family_has_nonzero_phi1(self) -> None:
        adj = _disjoint_cycles(10, 4)
        ratios = order_four_ratios(adj)
        assert ratios["phi_4_1"] > 0.0

    @pytest.mark.parametrize("cycle_len", [5, 6])
    def test_longer_cycles_have_zero_phi1(self, cycle_len: int) -> None:
        adj = _disjoint_cycles(10, cycle_len)
        ratios = order_four_ratios(adj)
        assert ratios["phi_4_1"] == 0.0

    def test_null_single_long_ring_has_zero_phi1(self) -> None:
        adj = _cycle_n(40)
        ratios = order_four_ratios(adj)
        assert ratios["phi_4_1"] == 0.0

    def test_families_all_degree_two_and_zero_clustering(self) -> None:
        """Sanity: the family really is matched on degree and clustering."""
        from craeft.graphs.metrics import global_clustering_coefficient  # noqa: PLC0415

        for adj in (
            _cycle_n(40),
            _disjoint_cycles(10, 4),
            _disjoint_cycles(10, 5),
            _disjoint_cycles(10, 6),
        ):
            degrees = np.asarray(adj.sum(axis=1)).ravel()
            assert np.all(degrees == 2.0)
            assert global_clustering_coefficient(adj) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# 6. Cross-check the induced 4-cycle count against an independent brute-force
#    counter (dev/control_audit.py's own chordless-cycle counter is out of
#    scope here since dev/ is gitignored; this reproduces the same check).
# ---------------------------------------------------------------------------


def _brute_force_induced_4cycles(adjacency: csr_matrix) -> int:
    """Reference count of induced (chordless) 4-cycles via exhaustive search."""
    a = adjacency.toarray() > 0
    n = a.shape[0]
    count = 0
    for combo in itertools.combinations(range(n), 4):
        edges = sum(
            1
            for i, j in itertools.combinations(combo, 2)
            if a[i, j]
        )
        if edges != 4:
            continue
        degs = sorted(
            sum(1 for j in combo if j != i and a[i, j]) for i in combo
        )
        if degs == [2, 2, 2, 2]:
            count += 1
    return count


class TestAgreesWithBruteForceCycleCounter:
    @pytest.mark.parametrize(
        "adj",
        [
            _cycle_n(12),
            _disjoint_cycles(3, 4),
            _disjoint_cycles(2, 5),
            _complete4(),
            _diamond4(),
        ],
    )
    def test_matches_brute_force(self, adj: csr_matrix) -> None:
        counts = count_order_four(adj)
        assert counts["cycle"] == _brute_force_induced_4cycles(adj)


# ---------------------------------------------------------------------------
# order_four_per_node
# ---------------------------------------------------------------------------


class TestOrderFourPerNode:
    def test_shape_matches_n_nodes(self) -> None:
        adj = _disjoint_cycles(3, 4)
        result = order_four_per_node(adj)
        for arr in result.values():
            assert arr.shape == (12,)

    def test_sum_matches_global_count_times_class_size(self) -> None:
        adj = _disjoint_cycles(3, 4)
        per_node = order_four_per_node(adj)
        global_counts = count_order_four(adj)
        for cls in ("path", "star", "cycle", "diamond", "complete"):
            assert per_node[cls].sum() == 4 * global_counts[cls]

    def test_single_cycle_all_nodes_participate_once(self) -> None:
        adj = _cycle4()
        per_node = order_four_per_node(adj)
        np.testing.assert_array_equal(per_node["cycle"], [1, 1, 1, 1])
        np.testing.assert_array_equal(per_node["diamond"], [0, 0, 0, 0])

    def test_single_diamond_all_four_nodes_participate_once(self) -> None:
        adj = _diamond4()
        per_node = order_four_per_node(adj)
        np.testing.assert_array_equal(per_node["diamond"], [1, 1, 1, 1])

    def test_shared_hub_participates_in_both_instances(self) -> None:
        """Two diamonds sharing node 1: node 1 gets counted twice.

        Diamond A on {0,1,2,3} (hubs 0,2), diamond B on {1,4,5,6} (hubs 1,5),
        sharing node 1 as a tip of A and a hub of B.
        """
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0), (0, 2),  # diamond A
            (1, 4), (4, 5), (5, 6), (6, 1), (1, 5),  # diamond B
        ]
        adj = _from_edges(7, edges)
        per_node = order_four_per_node(adj)
        assert per_node["diamond"][1] == 2
        assert per_node["diamond"][0] == 1
        assert per_node["diamond"][4] == 1
