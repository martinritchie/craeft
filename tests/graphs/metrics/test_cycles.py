"""Tests for induced (chordless) cycle counting metrics."""

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix

from craeft.graphs.metrics import (
    cycles_per_node,
    induced_cycle_count,
    triangles_per_node,
)


def _ring(n: int) -> csr_matrix:
    """Adjacency matrix of a single n-node cycle."""
    dense = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        dense[i, (i + 1) % n] = 1
        dense[(i + 1) % n, i] = 1
    return csr_matrix(dense)


def _random_graph(n: int, p: float, seed: int) -> csr_matrix:
    """Seeded Erdos-Renyi adjacency matrix."""
    rng = np.random.default_rng(seed)
    upper = rng.random((n, n)) < p
    dense = np.triu(upper, k=1)
    dense = dense | dense.T
    return csr_matrix(dense.astype(np.int8))


class TestInducedCycleCount:
    """Tests for the global induced-cycle counter."""

    def test_triangle_has_one_three_cycle(self, triangle_graph: csr_array) -> None:
        assert induced_cycle_count(triangle_graph, 3) == 1

    def test_complete_k4_has_four_three_cycles(
        self, complete_graph_k4: csr_array
    ) -> None:
        assert induced_cycle_count(complete_graph_k4, 3) == 4

    def test_complete_k4_has_no_induced_four_cycles(
        self, complete_graph_k4: csr_array
    ) -> None:
        assert induced_cycle_count(complete_graph_k4, 4) == 0

    def test_ring_of_five(self) -> None:
        assert induced_cycle_count(_ring(5), 5) == 1

    def test_ring_of_five_has_no_shorter_cycles(self) -> None:
        assert induced_cycle_count(_ring(5), 3) == 0
        assert induced_cycle_count(_ring(5), 4) == 0

    def test_square_has_one_induced_four_cycle(self) -> None:
        assert induced_cycle_count(_ring(4), 4) == 1

    def test_induced_excludes_chorded(
        self, two_triangles_shared_edge: csr_array
    ) -> None:
        """A diamond (K4 minus an edge) has zero induced 4-cycles."""
        assert induced_cycle_count(two_triangles_shared_edge, 4) == 0
        # ...but does contain the two chorded triangles.
        assert induced_cycle_count(two_triangles_shared_edge, 3) == 2

    def test_empty_graph(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.int8)
        assert induced_cycle_count(adj, 3) == 0
        assert induced_cycle_count(adj, 5) == 0

    def test_chain_has_no_cycles(self, chain_graph: csr_array) -> None:
        assert induced_cycle_count(chain_graph, 3) == 0
        assert induced_cycle_count(chain_graph, 4) == 0

    def test_matches_triangle_count_at_length_three(self) -> None:
        for seed in (0, 1, 2, 3):
            adj = _random_graph(12, 0.35, seed)
            expected = int(triangles_per_node(adj).sum()) // 3
            assert induced_cycle_count(adj, 3) == expected

    def test_length_below_three_raises(self, triangle_graph: csr_array) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            induced_cycle_count(triangle_graph, 2)


class TestCyclesPerNode:
    """Tests for the per-node induced-cycle counter."""

    def test_cycles_per_node_ring(self) -> None:
        """A single C5: every node sits on exactly one induced 5-cycle."""
        result = cycles_per_node(_ring(5), 5)
        np.testing.assert_array_equal(result, [1, 1, 1, 1, 1])
        assert result.sum() == 5

    def test_cycles_per_node_matches_triangles(self) -> None:
        """At length 3 this generalises `triangles_per_node` exactly."""
        for seed in (0, 1, 2, 3, 4):
            adj = _random_graph(14, 0.3, seed)
            np.testing.assert_array_equal(
                cycles_per_node(adj, 3), triangles_per_node(adj)
            )

    def test_cycles_per_node_sum_identity(self) -> None:
        """Each induced cycle contributes `length` node-increments."""
        for seed in (0, 1, 2, 3, 4):
            adj = _random_graph(12, 0.3, seed)
            for length in (4, 5):
                per_node = cycles_per_node(adj, length)
                assert per_node.sum() == length * induced_cycle_count(adj, length)

    def test_diamond_has_no_induced_four_cycles_per_node(
        self, two_triangles_shared_edge: csr_array
    ) -> None:
        result = cycles_per_node(two_triangles_shared_edge, 4)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_empty_graph(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.int8)
        np.testing.assert_array_equal(cycles_per_node(adj, 3), np.zeros(5))

    def test_star_has_no_cycles(self, star_graph: csr_array) -> None:
        np.testing.assert_array_equal(cycles_per_node(star_graph, 3), [0, 0, 0, 0])

    def test_returns_integer_array(self, triangle_graph: csr_array) -> None:
        assert np.issubdtype(cycles_per_node(triangle_graph, 3).dtype, np.integer)

    def test_length_below_three_raises(self, triangle_graph: csr_array) -> None:
        with pytest.raises(ValueError, match="at least 3"):
            cycles_per_node(triangle_graph, 2)


class TestSparseMatrixCompatibility:
    """Both counters accept the sparse formats used elsewhere in the package."""

    def test_csr_array_works(self, triangle_graph: csr_array) -> None:
        assert induced_cycle_count(triangle_graph, 3) == 1

    def test_csr_matrix_works(self, triangle_graph: csr_array) -> None:
        adj = csr_matrix(triangle_graph)
        assert induced_cycle_count(adj, 3) == 1
