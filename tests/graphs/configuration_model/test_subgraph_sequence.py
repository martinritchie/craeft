"""Tests for SubgraphSequence orbit properties."""

import numpy as np
import pytest
from scipy.stats import poisson

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model.sequence import (
    SubgraphSequence,
    split_by_degree_rank,
    split_deterministic,
)

# -- Test subgraphs ----------------------------------------------------------

TRIANGLE = Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))

SQUARE = Subgraph(
    adjacency=np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
)

K4 = Subgraph(adjacency=np.ones((4, 4), dtype=int) - np.eye(4, dtype=int))

# K4 minus edge 1-3: nodes 0,2 have degree 3; nodes 1,3 have degree 2.
DIAMOND = Subgraph(
    adjacency=np.array([[0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]])
)


def _seq(subgraph: Subgraph) -> SubgraphSequence:
    """Build a SubgraphSequence with a dummy distribution."""
    return SubgraphSequence(subgraph=subgraph, distribution=poisson(1))


# -- Orbits -------------------------------------------------------------------


class TestOrbitsTransitive:
    """Vertex-transitive subgraphs have a single orbit label."""

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4])
    def test_single_orbit_value(self, subgraph: Subgraph) -> None:
        assert len(set(_seq(subgraph).orbits)) == 1

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4])
    def test_orbit_label_is_zero(self, subgraph: Subgraph) -> None:
        assert set(_seq(subgraph).orbits) == {0}

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4])
    def test_length_matches_num_nodes(self, subgraph: Subgraph) -> None:
        assert len(_seq(subgraph).orbits) == subgraph.num_nodes


class TestOrbitsDiamond:
    """Diamond graph (K4 minus one edge) has two orbits."""

    def test_two_distinct_orbits(self) -> None:
        assert len(set(_seq(DIAMOND).orbits)) == 2

    def test_length_matches_num_nodes(self) -> None:
        assert len(_seq(DIAMOND).orbits) == DIAMOND.num_nodes

    def test_high_degree_nodes_share_orbit(self) -> None:
        """Nodes 0 and 2 (degree 3) are in the same orbit."""
        orbits = _seq(DIAMOND).orbits
        assert orbits[0] == orbits[2]

    def test_low_degree_nodes_share_orbit(self) -> None:
        """Nodes 1 and 3 (degree 2) are in the same orbit."""
        orbits = _seq(DIAMOND).orbits
        assert orbits[1] == orbits[3]

    def test_different_degree_nodes_differ(self) -> None:
        """Degree-3 and degree-2 nodes are in different orbits."""
        orbits = _seq(DIAMOND).orbits
        assert orbits[0] != orbits[1]


class TestOrbitsLabelling:
    """Orbit labels are consecutive integers from 0."""

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4, DIAMOND])
    def test_labels_start_at_zero(self, subgraph: Subgraph) -> None:
        assert min(_seq(subgraph).orbits) == 0

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4, DIAMOND])
    def test_labels_are_consecutive(self, subgraph: Subgraph) -> None:
        orbits = _seq(subgraph).orbits
        labels = set(orbits)
        assert labels == set(range(len(labels)))


# -- Orbit degrees ------------------------------------------------------------


class TestOrbitDegrees:
    """Mapping from orbit label to subgraph degree."""

    def test_triangle_all_degree_two(self) -> None:
        assert _seq(TRIANGLE).orbit_degrees == {0: 2}

    def test_square_all_degree_two(self) -> None:
        assert _seq(SQUARE).orbit_degrees == {0: 2}

    def test_k4_all_degree_three(self) -> None:
        assert _seq(K4).orbit_degrees == {0: 3}

    def test_diamond_two_orbit_degrees(self) -> None:
        od = _seq(DIAMOND).orbit_degrees
        assert set(od.values()) == {2, 3}
        assert len(od) == 2

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4, DIAMOND])
    def test_keys_match_orbit_labels(self, subgraph: Subgraph) -> None:
        seq = _seq(subgraph)
        assert set(seq.orbit_degrees.keys()) == set(seq.orbits)


# -- Orbit sizes --------------------------------------------------------------


class TestOrbitSizes:
    """Mapping from orbit label to vertex count."""

    def test_triangle_single_orbit_of_three(self) -> None:
        assert _seq(TRIANGLE).orbit_sizes == {0: 3}

    def test_square_single_orbit_of_four(self) -> None:
        assert _seq(SQUARE).orbit_sizes == {0: 4}

    def test_k4_single_orbit_of_four(self) -> None:
        assert _seq(K4).orbit_sizes == {0: 4}

    def test_diamond_two_orbits_of_two(self) -> None:
        sizes = _seq(DIAMOND).orbit_sizes
        assert sorted(sizes.values()) == [2, 2]

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4, DIAMOND])
    def test_sizes_sum_to_num_nodes(self, subgraph: Subgraph) -> None:
        assert sum(_seq(subgraph).orbit_sizes.values()) == subgraph.num_nodes


# -- Vertex transitivity ------------------------------------------------------


class TestIsVertexTransitive:
    """Boolean check for single-orbit subgraphs."""

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4])
    def test_transitive_subgraphs(self, subgraph: Subgraph) -> None:
        assert _seq(subgraph).is_vertex_transitive is True

    def test_diamond_is_not_transitive(self) -> None:
        assert _seq(DIAMOND).is_vertex_transitive is False


# -- edges_for -------------------------------------------------------------


class TestEdgesFor:
    """Mapping subgraph adjacency to concrete node IDs."""

    def test_triangle_on_three_nodes(self) -> None:
        nodes = np.array([5, 10, 15], dtype=np.int_)
        rows, cols = _seq(TRIANGLE).edges_for(nodes)
        edges = set((min(r, c), max(r, c)) for r, c in zip(rows, cols))
        assert edges == {(5, 10), (5, 15), (10, 15)}

    def test_square_on_four_nodes(self) -> None:
        nodes = np.array([0, 1, 2, 3], dtype=np.int_)
        rows, cols = _seq(SQUARE).edges_for(nodes)
        edges = set((min(r, c), max(r, c)) for r, c in zip(rows, cols))
        # Square: edges (0,1), (1,2), (2,3), (3,0)
        assert edges == {(0, 1), (1, 2), (2, 3), (0, 3)}

    def test_diamond_on_concrete_nodes(self) -> None:
        nodes = np.array([7, 8, 9, 10], dtype=np.int_)
        rows, cols = _seq(DIAMOND).edges_for(nodes)
        edges = set((min(r, c), max(r, c)) for r, c in zip(rows, cols))
        # diamond: (0,1), (0,2), (0,3), (1,2), (2,3)
        assert edges == {(7, 8), (7, 9), (7, 10), (8, 9), (9, 10)}

    def test_edge_count_matches_subgraph(self) -> None:
        nodes = np.array([0, 1, 2, 3], dtype=np.int_)
        rows, cols = _seq(K4).edges_for(nodes)
        assert len(rows) == K4.num_edges  # K4 has 6 edges


# -- split_by_orbit --------------------------------------------------------


class TestSplitByOrbitVertexTransitive:
    """Vertex-transitive subgraphs return single-orbit decomposition."""

    def test_triangle_returns_single_orbit(self) -> None:
        seq = _seq(TRIANGLE)
        rng = np.random.default_rng(42)
        # 2 instances = 6 participations total
        parts = np.array([1, 1, 1, 1, 1, 1], dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        assert set(result.keys()) == {0}
        assert len(result[0]) == 6
        assert result[0].sum() == 6

    def test_square_returns_single_orbit(self) -> None:
        seq = _seq(SQUARE)
        rng = np.random.default_rng(42)
        # 1 instance = 4 participations
        parts = np.array([1, 1, 1, 1], dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        assert set(result.keys()) == {0}

    def test_row_sums_unchanged(self) -> None:
        """Each node's total across orbits equals its participation count."""
        seq = _seq(TRIANGLE)
        rng = np.random.default_rng(42)
        parts = np.array([2, 1, 0, 3, 1, 2], dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        assert np.array_equal(result[0], parts)


class TestSplitByOrbitDiamond:
    """Diamond (two orbits) uses sequential urn sampling."""

    def test_returns_both_orbits(self) -> None:
        seq = _seq(DIAMOND)
        rng = np.random.default_rng(42)
        # 2 instances = 8 participations total
        parts = np.array([2, 2, 2, 2], dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        assert set(result.keys()) == {0, 1}

    def test_column_sums_match_orbit_proportions(self) -> None:
        """Total per-orbit counts must match M * σ_o."""
        seq = _seq(DIAMOND)
        rng = np.random.default_rng(42)
        n = 10
        # Poisson samples: total divisible by 4
        parts = np.full(n, 2, dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        # M = 20/4 = 5 instances, need 5*2 = 10 hub slots, 5*2 = 10 leaf slots
        assert result[0].sum() == 10
        assert result[1].sum() == 10

    def test_row_sums_match_participation(self) -> None:
        seq = _seq(DIAMOND)
        rng = np.random.default_rng(42)
        parts = np.array([3, 2, 2, 1], dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        for i in range(4):
            assert result[0][i] + result[1][i] == parts[i]

    def test_produces_integer_counts(self) -> None:
        seq = _seq(DIAMOND)
        rng = np.random.default_rng(42)
        parts = np.array([2, 2, 2, 2], dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        for arr in result.values():
            assert np.all(arr >= 0)
            assert np.all(arr == arr.astype(int))

    def test_rejects_infeasible_sequence(self) -> None:
        """Edge case: participation sum not divisible by num_nodes."""
        seq = _seq(DIAMOND)
        rng = np.random.default_rng(42)
        parts = np.array([1, 1, 1, 0], dtype=np.int_)  # sum=3, not div by 4
        with pytest.raises(ValueError):
            seq._split_by_orbit(parts, rng)


# -- Prescribed orbit_counts (ticket 007) -------------------------------------


class TestPrescribedOrbitCountsVerbatim:
    """A prescribed decomposition bypasses sampling entirely."""

    def test_prescribed_orbit_counts_used_verbatim(self) -> None:
        counts = {
            0: np.array([2, 1, 1, 0], dtype=np.int_),
            1: np.array([1, 1, 1, 1], dtype=np.int_),
        }
        seq = SubgraphSequence(
            subgraph=DIAMOND, distribution=poisson(1), orbit_counts=counts
        )
        rng = np.random.default_rng(0)
        # Deliberately inconsistent with `counts` — proves it's ignored.
        dummy_parts = np.array([9, 9, 9, 9], dtype=np.int_)

        result = seq._split_by_orbit(dummy_parts, rng)

        np.testing.assert_array_equal(result[0], counts[0])
        np.testing.assert_array_equal(result[1], counts[1])


class TestPrescribedOrbitCountsValidation:
    """__post_init__ validates orbit_counts eagerly."""

    def test_prescribed_counts_reject_inconsistent_totals(self) -> None:
        # sigma_0 = sigma_1 = 2 for the diamond. total_0=4 -> M=2,
        # total_1=2 -> M=1: no common M.
        counts = {
            0: np.array([2, 2, 0, 0], dtype=np.int_),
            1: np.array([1, 1, 0, 0], dtype=np.int_),
        }
        with pytest.raises(ValueError, match="inconsistent"):
            SubgraphSequence(
                subgraph=DIAMOND, distribution=poisson(1), orbit_counts=counts
            )

    def test_prescribed_counts_reject_wrong_keys(self) -> None:
        counts = {
            0: np.array([1, 1, 1, 1], dtype=np.int_),
            2: np.array([1, 1, 1, 1], dtype=np.int_),  # diamond has {0, 1}
        }
        with pytest.raises(ValueError, match="keys"):
            SubgraphSequence(
                subgraph=DIAMOND, distribution=poisson(1), orbit_counts=counts
            )

    def test_prescribed_counts_reject_negative(self) -> None:
        counts = {
            0: np.array([-1, 3, 0, 0], dtype=np.int_),
            1: np.array([1, 1, 0, 0], dtype=np.int_),
        }
        with pytest.raises(ValueError, match="negative"):
            SubgraphSequence(
                subgraph=DIAMOND, distribution=poisson(1), orbit_counts=counts
            )

    def test_prescribed_counts_reject_length_mismatch(self) -> None:
        counts = {
            0: np.array([1, 1, 1], dtype=np.int_),
            1: np.array([1, 1], dtype=np.int_),
        }
        with pytest.raises(ValueError, match="length"):
            SubgraphSequence(
                subgraph=DIAMOND, distribution=poisson(1), orbit_counts=counts
            )


class TestPrescribedSplitDeterminism:
    """The point of the ticket: identical prescription -> identical
    per-node designed triangle counts, regardless of seed."""

    def test_prescribed_split_gives_deterministic_local_clustering(self) -> None:
        # Diamond: each hub (orbit 0) participation sits in 2 triangles
        # per instance; each leaf (orbit 1) participation sits in 1.
        counts = {
            0: np.array([2, 1, 1, 0], dtype=np.int_),
            1: np.array([1, 1, 1, 1], dtype=np.int_),
        }
        seq = SubgraphSequence(
            subgraph=DIAMOND, distribution=poisson(1), orbit_counts=counts
        )
        dummy_parts = np.array([3, 2, 2, 1], dtype=np.int_)

        def designed_triangles(decomp: dict[int, np.ndarray]) -> np.ndarray:
            return 2 * decomp[0] + 1 * decomp[1]

        results = [
            designed_triangles(
                seq._split_by_orbit(dummy_parts, np.random.default_rng(seed))
            )
            for seed in range(8)
        ]
        for r in results[1:]:
            np.testing.assert_array_equal(r, results[0])


class TestSplitDeterministic:
    """Largest-remainder split: equal participation -> equal orbit counts."""

    def test_deterministic_split_equal_participation_equal_orbits(self) -> None:
        seq = _seq(DIAMOND)
        parts = np.array([2, 2, 2, 2], dtype=np.int_)  # all nodes equal
        result = split_deterministic(parts, seq)

        assert len(set(result[0].tolist())) == 1
        assert len(set(result[1].tolist())) == 1

        # Row sums preserved.
        np.testing.assert_array_equal(result[0] + result[1], parts)

        # Column sums exactly M * sigma_o.
        total = int(parts.sum())
        m = total // DIAMOND.num_nodes
        assert result[0].sum() == m * seq.orbit_sizes[0]
        assert result[1].sum() == m * seq.orbit_sizes[1]

    def test_row_and_column_sums_exact_for_unequal_participation(self) -> None:
        seq = _seq(DIAMOND)
        parts = np.array([3, 2, 2, 1], dtype=np.int_)
        result = split_deterministic(parts, seq)

        np.testing.assert_array_equal(result[0] + result[1], parts)
        total = int(parts.sum())
        m = total // DIAMOND.num_nodes
        assert result[0].sum() == m * seq.orbit_sizes[0]
        assert result[1].sum() == m * seq.orbit_sizes[1]

    def test_output_is_valid_prescribed_orbit_counts(self) -> None:
        """split_deterministic's output must pass SubgraphSequence's
        own orbit_counts validation — round-trip sanity check."""
        seq = _seq(DIAMOND)
        parts = np.array([3, 2, 2, 1], dtype=np.int_)
        result = split_deterministic(parts, seq)
        # Should not raise.
        SubgraphSequence(subgraph=DIAMOND, distribution=poisson(1), orbit_counts=result)


class TestSplitByDegreeRank:
    """Assigns high-cardinality (hub) orbits to high-degree nodes."""

    def test_degree_rank_split_raises_assortativity(self) -> None:
        """Pairs with ticket 005 (assortativity metric, not yet built):
        pushing the higher-degree orbit role onto high-degree nodes is
        exactly the construction that raises degree assortativity in
        the 2017 paper. This checks the underlying signature directly —
        a strong positive correlation between node degree and the
        stub-cost contributed by the (higher-cardinality) hub orbit —
        without depending on ticket 005's not-yet-implemented metric.
        """
        seq = _seq(DIAMOND)
        n = 8
        parts = np.full(n, 2, dtype=np.int_)
        degrees = np.array([10, 9, 8, 7, 6, 5, 4, 3], dtype=np.int_)

        result = split_by_degree_rank(parts, degrees, seq)
        hub_cost = result[0] * seq.orbit_degrees[0]

        corr = np.corrcoef(degrees, hub_cost)[0, 1]
        assert corr > 0.8

    def test_row_and_column_sums_exact(self) -> None:
        seq = _seq(DIAMOND)
        parts = np.array([3, 2, 2, 1], dtype=np.int_)
        degrees = np.array([9, 7, 5, 3], dtype=np.int_)
        result = split_by_degree_rank(parts, degrees, seq)

        np.testing.assert_array_equal(result[0] + result[1], parts)
        total = int(parts.sum())
        m = total // DIAMOND.num_nodes
        assert result[0].sum() == m * seq.orbit_sizes[0]
        assert result[1].sum() == m * seq.orbit_sizes[1]

    def test_output_is_valid_prescribed_orbit_counts(self) -> None:
        seq = _seq(DIAMOND)
        parts = np.array([3, 2, 2, 1], dtype=np.int_)
        degrees = np.array([9, 7, 5, 3], dtype=np.int_)
        result = split_by_degree_rank(parts, degrees, seq)
        # Should not raise.
        SubgraphSequence(subgraph=DIAMOND, distribution=poisson(1), orbit_counts=result)


class TestVertexTransitiveUnaffected:
    """Regression: vertex-transitive subgraphs unaffected by orbit_counts."""

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4])
    def test_no_orbit_counts_still_returns_sequence_verbatim(
        self, subgraph: Subgraph
    ) -> None:
        seq = _seq(subgraph)
        rng = np.random.default_rng(1)
        parts = np.array([2, 1, 0, 3][: subgraph.num_nodes], dtype=np.int_)
        result = seq._split_by_orbit(parts, rng)
        assert set(result.keys()) == {0}
        np.testing.assert_array_equal(result[0], parts)

    def test_prescribed_counts_bypass_even_when_vertex_transitive(self) -> None:
        counts = {0: np.array([2, 1, 0], dtype=np.int_)}  # sum=3, sigma_0=3
        seq = SubgraphSequence(
            subgraph=TRIANGLE, distribution=poisson(1), orbit_counts=counts
        )
        assert seq.is_vertex_transitive
        rng = np.random.default_rng(5)
        dummy_parts = np.array([9, 9, 9], dtype=np.int_)

        result = seq._split_by_orbit(dummy_parts, rng)

        np.testing.assert_array_equal(result[0], counts[0])

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4])
    def test_split_deterministic_returns_sequence_verbatim(
        self, subgraph: Subgraph
    ) -> None:
        seq = _seq(subgraph)
        parts = np.array([2, 1, 0, 3][: subgraph.num_nodes], dtype=np.int_)
        result = split_deterministic(parts, seq)
        assert set(result.keys()) == {0}
        np.testing.assert_array_equal(result[0], parts)

    @pytest.mark.parametrize("subgraph", [TRIANGLE, SQUARE, K4])
    def test_split_by_degree_rank_returns_sequence_verbatim(
        self, subgraph: Subgraph
    ) -> None:
        seq = _seq(subgraph)
        parts = np.array([2, 1, 0, 3][: subgraph.num_nodes], dtype=np.int_)
        degrees = np.array([9, 7, 5, 3][: subgraph.num_nodes], dtype=np.int_)
        result = split_by_degree_rank(parts, degrees, seq)
        assert set(result.keys()) == {0}
        np.testing.assert_array_equal(result[0], parts)


# -- distribution is optional when fully prescribed (ticket 007, gap 3) -------


class TestDistributionOptional:
    """A fully prescribed sequence has nothing left to sample, so it
    must not be forced to carry a participation distribution."""

    PRESCRIBED = {
        0: np.array([2, 1, 1, 0], dtype=np.int_),  # sum=4, sigma_0=2 -> M=2
        1: np.array([1, 1, 1, 1], dtype=np.int_),  # sum=4, sigma_1=2 -> M=2
    }

    def test_constructs_without_distribution(self) -> None:
        seq = SubgraphSequence(subgraph=DIAMOND, orbit_counts=self.PRESCRIBED)
        assert seq.distribution is None

    def test_neither_distribution_nor_orbit_counts_raises(self) -> None:
        with pytest.raises(ValueError, match="distribution"):
            SubgraphSequence(subgraph=DIAMOND)

    def test_sample_without_distribution_raises(self) -> None:
        seq = SubgraphSequence(subgraph=DIAMOND, orbit_counts=self.PRESCRIBED)
        with pytest.raises(ValueError, match="distribution"):
            seq.sample(4, np.random.default_rng(0))

    def test_split_by_orbit_works_without_distribution(self) -> None:
        seq = SubgraphSequence(subgraph=DIAMOND, orbit_counts=self.PRESCRIBED)
        result = seq._split_by_orbit(
            np.zeros(4, dtype=np.int_), np.random.default_rng(0)
        )
        np.testing.assert_array_equal(result[0], self.PRESCRIBED[0])
        np.testing.assert_array_equal(result[1], self.PRESCRIBED[1])


class TestNumInstances:
    """A prescription pins the instance count M exactly; a sampled
    sequence only knows it in expectation, so reports None."""

    def test_exact_when_prescribed(self) -> None:
        counts = {
            0: np.array([2, 1, 1, 0], dtype=np.int_),
            1: np.array([1, 1, 1, 1], dtype=np.int_),
        }
        seq = SubgraphSequence(subgraph=DIAMOND, orbit_counts=counts)
        assert seq.num_instances == 2

    def test_exact_when_prescribed_vertex_transitive(self) -> None:
        counts = {0: np.array([2, 1, 3], dtype=np.int_)}  # sum=6, sigma_0=3
        seq = SubgraphSequence(subgraph=TRIANGLE, orbit_counts=counts)
        assert seq.num_instances == 2

    def test_none_when_sampled(self) -> None:
        assert _seq(DIAMOND).num_instances is None
