"""Tests for SubgraphSequence orbit properties."""

import numpy as np
import pytest
from scipy.stats import poisson

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model.sequence import SubgraphSequence

# -- Test subgraphs ----------------------------------------------------------

TRIANGLE = Subgraph(
    adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])
)

SQUARE = Subgraph(
    adjacency=np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]])
)

K4 = Subgraph(
    adjacency=np.ones((4, 4), dtype=int) - np.eye(4, dtype=int)
)

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
