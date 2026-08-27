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

    Catalogued as G6 in `docs/concepts/motifs.md` and not named in the
    ticket's census table, but a real 6th connected
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
            adjacency=np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]])
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
    """phi_4^2 + phi_4^3 + phi_4^4 + (1 - phi_4^1) == 1.

    Two judgment calls documented here. **The first was inverted until
    ticket 008 and is corrected below; the second stands, with a better
    justification than the one originally given.**

    1. **phi_4^2 is the paper's own name for the empty square, and phi_4^1
       is the aggregate.** This docstring previously read the ticket's
       phi_4^2 term as "a typo carried over from the paper's original
       numbering", reasoning that there are exactly three isomorphism
       classes built from "a square plus 0/1/2 diagonals" so there is no
       room for a fourth. The reasoning is sound but answers the wrong
       question: the fourth symbol is not a fourth class, it is the
       *aggregate* of the other three. The published PDF (p. 24 Section 2.2
       item 4; Table 2, p. 28) names phi_4^1 = all closed quadruples,
       phi_4^2 = empty square, phi_4^3 = diamond, phi_4^4 = K4, and
       Table 2 satisfies phi_4^1 = phi_4^2 + phi_4^3 + phi_4^4 row by row.
       The typo was ours -- a pdftotext transcription dropped a superscript
       -- not the paper's. See `TestAggregateIsAdditive` below.

    2. **The paw belongs in `unclosed`** -- kept, but not because it is
       "not one of the three named closed classes". The paper defines these
       ratios over *"4-node structures connected in a loop"*, and a paw
       contains no 4-cycle, so it is unclosed **by the paper's own
       definition**, not by a choice this library makes. It is additionally
       reported under its own `paw` key (ticket 008) as a diagnostic, but
       that key is a *component* of `unclosed`, not a further term: the four
       terms below still sum to exactly 1 without it.
    """

    @pytest.mark.parametrize("builder", CLASS_BUILDERS.values())
    def test_single_instance(self, builder) -> None:
        ratios = order_four_ratios(builder())
        total = (
            ratios["phi_4_2"]
            + ratios["phi_4_3"]
            + ratios["phi_4_4"]
            + ratios["unclosed"]
        )
        assert total == pytest.approx(1.0)

    def test_mixed_population(self) -> None:
        adj = _disjoint_cycles(3, 4)
        ratios = order_four_ratios(adj)
        total = (
            ratios["phi_4_2"]
            + ratios["phi_4_3"]
            + ratios["phi_4_4"]
            + ratios["unclosed"]
        )
        assert total == pytest.approx(1.0)

    def test_empty_graph_is_zero_not_nan(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.float64)
        ratios = order_four_ratios(adj)
        assert set(ratios) == {
            "phi_4_1",
            "phi_4_2",
            "phi_4_3",
            "phi_4_4",
            "unclosed",
            "paw",
        }
        assert all(v == 0.0 for v in ratios.values())


# ---------------------------------------------------------------------------
# 4b. The corrected numbering: phi_4^1 is the aggregate (ticket 008)
# ---------------------------------------------------------------------------


def _mixed_population() -> csr_matrix:
    """One instance of every 4-node class, in disjoint components.

    Guarantees all six classes are present at once, so the aggregate
    identity is exercised with every term non-zero.
    """
    edges: list[tuple[int, int]] = []
    for i, builder in enumerate(CLASS_BUILDERS.values()):
        base = 4 * i
        block = builder().toarray() > 0
        edges += [
            (base + r, base + c)
            for r in range(4)
            for c in range(r + 1, 4)
            if block[r, c]
        ]
    return _from_edges(4 * len(CLASS_BUILDERS), edges)


class TestAggregateIsAdditive:
    """phi_4^1 == phi_4^2 + phi_4^3 + phi_4^4 -- the identity 2014 Table 2 satisfies.

    This is the test that pins the corrected numbering: it fails outright if
    phi_4_1 is ever moved back to naming the empty square.
    """

    def test_phi_4_1_is_aggregate_of_2_3_4(self) -> None:
        ratios = order_four_ratios(_mixed_population())
        assert ratios["phi_4_1"] > 0.0
        assert ratios["phi_4_1"] == pytest.approx(
            ratios["phi_4_2"] + ratios["phi_4_3"] + ratios["phi_4_4"]
        )

    @pytest.mark.parametrize("builder", CLASS_BUILDERS.values())
    def test_aggregate_additive_on_every_single_class(self, builder) -> None:
        ratios = order_four_ratios(builder())
        assert ratios["phi_4_1"] == pytest.approx(
            ratios["phi_4_2"] + ratios["phi_4_3"] + ratios["phi_4_4"]
        )

    def test_phi_4_1_plus_unclosed_is_one(self) -> None:
        for adj in (_mixed_population(), _disjoint_cycles(3, 4), _paw4()):
            ratios = order_four_ratios(adj)
            assert ratios["phi_4_1"] + ratios["unclosed"] == pytest.approx(1.0)

    def test_phi_4_2_is_the_empty_square(self) -> None:
        """The renamed key must carry the value phi_4_1 used to carry."""
        ratios = order_four_ratios(_cycle4())
        assert ratios["phi_4_2"] == pytest.approx(1.0)
        assert ratios["phi_4_3"] == 0.0
        assert ratios["phi_4_4"] == 0.0


class TestPawIsAComponentOfUnclosed:
    """`paw` is reported separately but is *inside* `unclosed`, not beside it."""

    @pytest.mark.parametrize("builder", CLASS_BUILDERS.values())
    def test_paw_bounded_by_unclosed(self, builder) -> None:
        ratios = order_four_ratios(builder())
        assert 0.0 <= ratios["paw"] <= ratios["unclosed"] + 1e-12

    def test_isolated_paw_is_wholly_unclosed(self) -> None:
        ratios = order_four_ratios(_paw4())
        assert ratios["paw"] == pytest.approx(1.0)
        assert ratios["unclosed"] == pytest.approx(1.0)
        assert ratios["phi_4_1"] == 0.0

    def test_paw_not_double_counted_into_any_phi(self) -> None:
        """Adding paws must move `unclosed` and `paw` only, never a phi_4^i."""
        base = _mixed_population()
        n_base = base.shape[0]
        extra_edges = [(r, c) for r, c in zip(*np.triu(base.toarray() > 0).nonzero())]
        for i in range(3):  # three more disjoint paws
            b = n_base + 4 * i
            extra_edges += [(b, b + 1), (b + 1, b + 2), (b + 2, b), (b + 2, b + 3)]
        with_paws = _from_edges(n_base + 12, extra_edges)

        before = order_four_ratios(base)
        after = order_four_ratios(with_paws)
        counts_before = count_order_four(base)
        counts_after = count_order_four(with_paws)

        assert counts_after["paw"] == counts_before["paw"] + 3
        for cls in ("cycle", "diamond", "complete"):
            assert counts_after[cls] == counts_before[cls]
        assert after["paw"] > before["paw"]
        assert after["unclosed"] > before["unclosed"]
        for key in ("phi_4_1", "phi_4_2", "phi_4_3", "phi_4_4"):
            assert after[key] < before[key]  # same numerator, larger denominator


class TestDenominatorIncludesStars:
    """Guards the denominator decision documented in the module docstring.

    craeft counts connected 4-node *induced subgraphs* (stars included); the
    paper's Appendix A.2 counts 4-node *paths*, which a star has none of.
    Under the path reading a lone star would have an empty denominator and
    every ratio would be undefined. Pinning the star's ratios here makes a
    future change to that choice fail loudly rather than silently shifting
    every phi_4^i by ~30%.
    """

    def test_single_star_gives_zero_closed_and_unclosed_one(self) -> None:
        ratios = order_four_ratios(_star4())
        assert ratios["phi_4_1"] == 0.0
        assert ratios["phi_4_2"] == 0.0
        assert ratios["phi_4_3"] == 0.0
        assert ratios["phi_4_4"] == 0.0
        assert ratios["paw"] == 0.0
        assert ratios["unclosed"] == pytest.approx(1.0)

    def test_stars_dilute_the_closed_ratios(self) -> None:
        """A square alone vs the same square beside a star."""
        alone = order_four_ratios(_cycle4())
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (4, 5), (4, 6), (4, 7)]
        beside_star = order_four_ratios(_from_edges(8, edges))
        assert alone["phi_4_2"] == pytest.approx(1.0)
        assert beside_star["phi_4_2"] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# 5. The recommended {Null, C4, C5, C6} family: phi_4^2 separates C4 (the
#    ticket's real purpose -- distinguishing networks matched on degree and
#    clustering by their order-four composition)
# ---------------------------------------------------------------------------


class TestEmptyCycleFamilyDiffersInOrderFour:
    def test_c4_family_has_nonzero_phi2(self) -> None:
        adj = _disjoint_cycles(10, 4)
        ratios = order_four_ratios(adj)
        assert ratios["phi_4_2"] > 0.0

    @pytest.mark.parametrize("cycle_len", [5, 6])
    def test_longer_cycles_have_zero_phi2(self, cycle_len: int) -> None:
        adj = _disjoint_cycles(10, cycle_len)
        ratios = order_four_ratios(adj)
        assert ratios["phi_4_2"] == 0.0

    def test_null_single_long_ring_has_zero_phi2(self) -> None:
        adj = _cycle_n(40)
        ratios = order_four_ratios(adj)
        assert ratios["phi_4_2"] == 0.0

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
        edges = sum(1 for i, j in itertools.combinations(combo, 2) if a[i, j])
        if edges != 4:
            continue
        degs = sorted(sum(1 for j in combo if j != i and a[i, j]) for i in combo)
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
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (0, 2),  # diamond A
            (1, 4),
            (4, 5),
            (5, 6),
            (6, 1),
            (1, 5),  # diamond B
        ]
        adj = _from_edges(7, edges)
        per_node = order_four_per_node(adj)
        assert per_node["diamond"][1] == 2
        assert per_node["diamond"][0] == 1
        assert per_node["diamond"][4] == 1
