"""Tests for clustering coefficient and triangle counting metrics."""

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix

from craeft.networks.metrics import (
    count_triangles,
    global_clustering_coefficient,
    local_clustering,
    triangles_per_node,
)


class TestCountTriangles:
    """Tests for triangle counting."""

    def test_single_triangle(self, triangle_graph: csr_array) -> None:
        assert count_triangles(triangle_graph) == 1

    def test_complete_k4_has_four_triangles(self, complete_graph_k4: csr_array) -> None:
        # K4 has C(4,3) = 4 triangles
        assert count_triangles(complete_graph_k4) == 4

    def test_chain_has_no_triangles(self, chain_graph: csr_array) -> None:
        assert count_triangles(chain_graph) == 0

    def test_star_has_no_triangles(self, star_graph: csr_array) -> None:
        assert count_triangles(star_graph) == 0

    def test_two_triangles_shared_edge(
        self, two_triangles_shared_edge: csr_array
    ) -> None:
        assert count_triangles(two_triangles_shared_edge) == 2

    def test_empty_graph(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.float64)
        assert count_triangles(adj) == 0


class TestTrianglesPerNode:
    """Tests for per-node triangle counts."""

    def test_single_triangle_all_nodes_have_one(
        self, triangle_graph: csr_array
    ) -> None:
        result = triangles_per_node(triangle_graph)
        np.testing.assert_array_equal(result, [1, 1, 1])

    def test_complete_k4_all_nodes_have_three(
        self, complete_graph_k4: csr_array
    ) -> None:
        # Each node in K4 participates in C(3,2) = 3 triangles
        result = triangles_per_node(complete_graph_k4)
        np.testing.assert_array_equal(result, [3, 3, 3, 3])

    def test_star_all_nodes_have_zero(self, star_graph: csr_array) -> None:
        result = triangles_per_node(star_graph)
        np.testing.assert_array_equal(result, [0, 0, 0, 0])

    def test_two_triangles_central_nodes_have_two(
        self, two_triangles_shared_edge: csr_array
    ) -> None:
        result = triangles_per_node(two_triangles_shared_edge)
        # Nodes 0 and 3 are in one triangle each; nodes 1 and 2 are in both
        np.testing.assert_array_equal(result, [1, 2, 2, 1])


class TestLocalClustering:
    """Tests for local clustering coefficient."""

    def test_triangle_all_ones(self, triangle_graph: csr_array) -> None:
        result = local_clustering(triangle_graph)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0])

    def test_complete_k4_all_ones(self, complete_graph_k4: csr_array) -> None:
        result = local_clustering(complete_graph_k4)
        np.testing.assert_array_almost_equal(result, [1.0, 1.0, 1.0, 1.0])

    def test_star_central_node_is_zero(self, star_graph: csr_array) -> None:
        result = local_clustering(star_graph)
        # Central node has 3 neighbors with no edges between them
        assert result[0] == 0.0

    def test_star_leaf_nodes_are_zero(self, star_graph: csr_array) -> None:
        result = local_clustering(star_graph)
        # Leaf nodes have degree 1, so clustering is 0 by definition
        np.testing.assert_array_equal(result[1:], [0.0, 0.0, 0.0])

    def test_chain_all_zeros(self, chain_graph: csr_array) -> None:
        result = local_clustering(chain_graph)
        np.testing.assert_array_equal(result, [0.0, 0.0, 0.0, 0.0])

    def test_two_triangles_shared_edge(
        self, two_triangles_shared_edge: csr_array
    ) -> None:
        result = local_clustering(two_triangles_shared_edge)
        # Nodes 0, 3: degree 2, 1 triangle -> C = 1.0
        # Nodes 1, 2: degree 3, 2 triangles -> C = 2 / 3 ≈ 0.667
        expected = [1.0, 2 / 3, 2 / 3, 1.0]
        np.testing.assert_array_almost_equal(result, expected)

    def test_isolated_nodes_have_zero_clustering(self) -> None:
        # Graph with one isolated node
        adj = csr_matrix(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 0],
                [1, 1, 0, 0],
                [0, 0, 0, 0],  # Isolated node
            ],
            dtype=np.float64,
        )
        result = local_clustering(adj)
        assert result[3] == 0.0


class TestGlobalClusteringCoefficient:
    """Tests for global clustering coefficient (transitivity)."""

    def test_triangle_is_one(self, triangle_graph: csr_array) -> None:
        result = global_clustering_coefficient(triangle_graph)
        assert result == pytest.approx(1.0)

    def test_complete_k4_is_one(self, complete_graph_k4: csr_array) -> None:
        result = global_clustering_coefficient(complete_graph_k4)
        assert result == pytest.approx(1.0)

    def test_star_is_zero(self, star_graph: csr_array) -> None:
        result = global_clustering_coefficient(star_graph)
        assert result == pytest.approx(0.0)

    def test_chain_is_zero(self, chain_graph: csr_array) -> None:
        result = global_clustering_coefficient(chain_graph)
        assert result == pytest.approx(0.0)

    def test_two_triangles_shared_edge(
        self, two_triangles_shared_edge: csr_array
    ) -> None:
        # 2 triangles = 6 closed triples
        # Connected triples: nodes 0,3 contribute 1 each, nodes 1,2 contribute 3 each
        # Total triples = 1 + 3 + 3 + 1 = 8
        # Global clustering = 6 / 8 = 0.75
        result = global_clustering_coefficient(two_triangles_shared_edge)
        assert result == pytest.approx(0.75)

    def test_empty_graph_is_zero(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.float64)
        result = global_clustering_coefficient(adj)
        assert result == pytest.approx(0.0)

    def test_single_edge_is_zero(self) -> None:
        # Two nodes connected by one edge: no triples possible
        adj = csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        result = global_clustering_coefficient(adj)
        assert result == pytest.approx(0.0)


class TestSparseMatrixCompatibility:
    """Tests for compatibility with different sparse matrix formats."""

    def test_csr_matrix_works(self, triangle_graph: csr_array) -> None:
        adj = csr_matrix(triangle_graph)
        assert count_triangles(adj) == 1
        assert global_clustering_coefficient(adj) == pytest.approx(1.0)

    def test_accepts_integer_dtype(self) -> None:
        adj = csr_matrix([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int32)
        assert count_triangles(adj) == 1
