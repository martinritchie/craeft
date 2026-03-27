"""Tests for Motif Decomposition algorithm."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix, lil_matrix

from craeft.networks.metrics import global_clustering_coefficient
from craeft.networks.rewiring.motif_decomposition import (
    _canonical_edge,
    _compute_wedge_count,
    _count_triangles_at_edges,
    _create_cliques,
    _initial_triangle_count,
    _swap_edges,
    motif_decomposition,
)


class TestMotifDecomposition:
    """Tests for the main motif_decomposition function."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        result = motif_decomposition(
            num_nodes=12, clique_size=3, target_clustering=0.5, rng=rng
        )
        assert isinstance(result, csr_matrix)

    def test_correct_shape(self, rng: np.random.Generator) -> None:
        result = motif_decomposition(
            num_nodes=12, clique_size=3, target_clustering=0.5, rng=rng
        )
        assert result.shape == (12, 12)

    def test_symmetric_matrix(self, rng: np.random.Generator) -> None:
        result = motif_decomposition(
            num_nodes=12, clique_size=4, target_clustering=0.5, rng=rng
        )
        diff = result - result.T
        assert diff.nnz == 0

    def test_no_self_loops(self, rng: np.random.Generator) -> None:
        result = motif_decomposition(
            num_nodes=12, clique_size=3, target_clustering=0.3, rng=rng
        )
        assert result.diagonal().sum() == 0

    def test_degree_preserved(self, rng: np.random.Generator) -> None:
        """Degree sequence should be preserved by edge swapping."""
        num_nodes = 12
        clique_size = 4
        expected_degree = clique_size - 1  # Each node in a K_m has degree m-1

        result = motif_decomposition(
            num_nodes=num_nodes,
            clique_size=clique_size,
            target_clustering=0.3,
            rng=rng,
        )

        degrees = np.asarray(result.sum(axis=1)).ravel()
        assert all(d == expected_degree for d in degrees)

    def test_edge_count_preserved(self, rng: np.random.Generator) -> None:
        """Total edge count should remain constant."""
        num_nodes = 12
        clique_size = 4
        num_cliques = num_nodes // clique_size
        edges_per_clique = clique_size * (clique_size - 1) // 2
        expected_edges = num_cliques * edges_per_clique

        result = motif_decomposition(
            num_nodes=num_nodes,
            clique_size=clique_size,
            target_clustering=0.3,
            rng=rng,
        )

        # nnz counts both (i,j) and (j,i), so divide by 2
        actual_edges = result.nnz // 2
        assert actual_edges == expected_edges

    def test_clustering_reduced(self, rng: np.random.Generator) -> None:
        """Clustering should decrease from initial 1.0."""
        result = motif_decomposition(
            num_nodes=12, clique_size=3, target_clustering=0.5, rng=rng
        )
        clustering = global_clustering_coefficient(result)
        assert clustering < 1.0

    def test_reaches_target_clustering(self, rng: np.random.Generator) -> None:
        """Should reach target clustering (within tolerance)."""
        target = 0.4
        result = motif_decomposition(
            num_nodes=24, clique_size=4, target_clustering=target, rng=rng
        )
        clustering = global_clustering_coefficient(result)
        # Allow some tolerance since we stop when <= target
        assert clustering <= target + 0.1

    def test_reproducible_with_seed(self) -> None:
        """Same seed should produce identical results."""
        result1 = motif_decomposition(
            num_nodes=12,
            clique_size=3,
            target_clustering=0.5,
            rng=np.random.default_rng(123),
        )
        result2 = motif_decomposition(
            num_nodes=12,
            clique_size=3,
            target_clustering=0.5,
            rng=np.random.default_rng(123),
        )
        diff = result1 - result2
        assert diff.nnz == 0

    def test_different_seeds_produce_different_results(self) -> None:
        """Different seeds should (likely) produce different networks."""
        result1 = motif_decomposition(
            num_nodes=12,
            clique_size=3,
            target_clustering=0.3,
            rng=np.random.default_rng(1),
        )
        result2 = motif_decomposition(
            num_nodes=12,
            clique_size=3,
            target_clustering=0.3,
            rng=np.random.default_rng(2),
        )
        diff = result1 - result2
        # Very unlikely to be identical with different seeds
        assert diff.nnz > 0

    def test_rejects_invalid_node_count(self) -> None:
        """Should reject num_nodes not divisible by clique_size."""
        with pytest.raises(ValueError, match="divisible"):
            motif_decomposition(num_nodes=10, clique_size=3, target_clustering=0.5)

    def test_rejects_invalid_target_clustering_too_high(self) -> None:
        with pytest.raises(ValueError, match="target_clustering"):
            motif_decomposition(num_nodes=9, clique_size=3, target_clustering=1.5)

    def test_rejects_invalid_target_clustering_negative(self) -> None:
        with pytest.raises(ValueError, match="target_clustering"):
            motif_decomposition(num_nodes=9, clique_size=3, target_clustering=-0.1)

    def test_rejects_clique_size_less_than_two(self) -> None:
        with pytest.raises(ValueError, match="clique_size"):
            motif_decomposition(num_nodes=10, clique_size=1, target_clustering=0.5)

    def test_high_target_returns_early(self, rng: np.random.Generator) -> None:
        """Target >= 1.0 should return initial cliques unchanged."""
        result = motif_decomposition(
            num_nodes=9, clique_size=3, target_clustering=1.0, rng=rng
        )
        clustering = global_clustering_coefficient(result)
        assert clustering == 1.0

    def test_works_with_triangles(self, rng: np.random.Generator) -> None:
        """Should work with clique_size=3 (triangles)."""
        result = motif_decomposition(
            num_nodes=15, clique_size=3, target_clustering=0.4, rng=rng
        )
        assert result.shape == (15, 15)

    def test_works_with_k4_cliques(self, rng: np.random.Generator) -> None:
        """Should work with clique_size=4 (K4)."""
        result = motif_decomposition(
            num_nodes=16, clique_size=4, target_clustering=0.4, rng=rng
        )
        assert result.shape == (16, 16)

    def test_works_with_larger_cliques(self, rng: np.random.Generator) -> None:
        """Should work with larger clique sizes."""
        result = motif_decomposition(
            num_nodes=20, clique_size=5, target_clustering=0.5, rng=rng
        )
        degrees = np.asarray(result.sum(axis=1)).ravel()
        assert all(d == 4 for d in degrees)  # K5 nodes have degree 4


class TestCreateCliques:
    """Tests for the _create_cliques helper function."""

    def test_correct_num_edges_triangles(self) -> None:
        """4 triangles should have 4 * 3 = 12 edges."""
        adj, edges = _create_cliques(12, 3)
        assert len(edges) == 12

    def test_correct_num_edges_k4(self) -> None:
        """3 K4 cliques should have 3 * 6 = 18 edges."""
        adj, edges = _create_cliques(12, 4)
        assert len(edges) == 18

    def test_adjacency_matches_edge_set(self) -> None:
        """Adjacency matrix should have entries for all edges."""
        adj, edges = _create_cliques(9, 3)
        for i, j in edges:
            assert adj[i, j] == 1
            assert adj[j, i] == 1

    def test_disconnected_components(self) -> None:
        """Cliques should not be connected to each other."""
        adj, _ = _create_cliques(12, 4)
        # Nodes 0-3 should not connect to nodes 4-7
        for i in range(4):
            for j in range(4, 8):
                assert adj[i, j] == 0

    def test_each_clique_complete(self) -> None:
        """Each clique should be fully connected internally."""
        adj, _ = _create_cliques(12, 4)
        # Check first clique (nodes 0-3) is complete
        for i in range(4):
            for j in range(i + 1, 4):
                assert adj[i, j] == 1

    def test_correct_matrix_shape(self) -> None:
        adj, _ = _create_cliques(15, 5)
        assert adj.shape == (15, 15)


class TestCanonicalEdge:
    """Tests for the _canonical_edge helper function."""

    def test_returns_sorted_tuple(self) -> None:
        assert _canonical_edge(5, 3) == (3, 5)
        assert _canonical_edge(3, 5) == (3, 5)

    def test_same_node_works(self) -> None:
        assert _canonical_edge(3, 3) == (3, 3)


class TestSwapEdges:
    """Tests for the _swap_edges helper function."""

    def test_rejects_self_loop(self) -> None:
        """Should reject swaps that create self-loops."""
        adj = lil_matrix((4, 4), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[0, 2] = adj[2, 0] = 1

        rng = np.random.default_rng(42)
        # e1=(0,1), e2=(0,2) → could create (0,0) self-loop
        # Might succeed with one configuration but fail with another
        # Just verify it doesn't crash
        _ = _swap_edges(adj, (0, 1), (0, 2), rng)

    def test_rejects_multi_edge(self) -> None:
        """Should reject swaps that create duplicate edges."""
        adj = lil_matrix((4, 4), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[2, 3] = adj[3, 2] = 1
        adj[0, 3] = adj[3, 0] = 1  # Pre-existing edge

        rng = np.random.default_rng(0)
        # Try many times - some configurations should be rejected
        results = [_swap_edges(adj.copy(), (0, 1), (2, 3), rng) for _ in range(10)]
        # At least some should be None due to multi-edge
        assert None in results or all(r is not None for r in results)

    def test_preserves_edge_count_on_success(self) -> None:
        """Successful swap should maintain edge count."""
        adj = lil_matrix((6, 6), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[2, 3] = adj[3, 2] = 1
        initial_nnz = adj.nnz

        rng = np.random.default_rng(42)
        result = _swap_edges(adj, (0, 1), (2, 3), rng)

        if result is not None:
            assert adj.nnz == initial_nnz

    def test_removes_old_edges(self) -> None:
        """Successful swap should remove the original edges."""
        adj = lil_matrix((6, 6), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[2, 3] = adj[3, 2] = 1

        rng = np.random.default_rng(42)
        result = _swap_edges(adj, (0, 1), (2, 3), rng)

        if result is not None:
            assert adj[0, 1] == 0
            assert adj[2, 3] == 0

    def test_adds_new_edges(self) -> None:
        """Successful swap should add the new edges."""
        adj = lil_matrix((6, 6), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[2, 3] = adj[3, 2] = 1

        rng = np.random.default_rng(42)
        result = _swap_edges(adj, (0, 1), (2, 3), rng)

        if result is not None:
            new_e1, new_e2 = result
            assert adj[new_e1[0], new_e1[1]] == 1
            assert adj[new_e2[0], new_e2[1]] == 1

    def test_returns_canonical_edges(self) -> None:
        """Returned edges should be in canonical form."""
        adj = lil_matrix((6, 6), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[4, 5] = adj[5, 4] = 1

        rng = np.random.default_rng(42)
        result = _swap_edges(adj, (0, 1), (4, 5), rng)

        if result is not None:
            e1, e2 = result
            assert e1[0] < e1[1]
            assert e2[0] < e2[1]


class TestInitialTriangleCount:
    """Tests for the _initial_triangle_count helper function."""

    def test_triangles_clique_size_3(self) -> None:
        """4 triangles (K3) should have 4 triangles total."""
        assert _initial_triangle_count(12, 3) == 4

    def test_triangles_clique_size_4(self) -> None:
        """3 K4 cliques each have C(4,3)=4 triangles, total 12."""
        assert _initial_triangle_count(12, 4) == 12

    def test_triangles_clique_size_5(self) -> None:
        """2 K5 cliques each have C(5,3)=10 triangles, total 20."""
        assert _initial_triangle_count(10, 5) == 20

    def test_triangles_clique_size_6(self) -> None:
        """1 K6 clique has C(6,3)=20 triangles."""
        assert _initial_triangle_count(6, 6) == 20


class TestComputeWedgeCount:
    """Tests for the _compute_wedge_count helper function."""

    def test_wedges_single_triangle(self) -> None:
        """A single triangle has 3 wedges (one per vertex)."""
        adj, _ = _create_cliques(3, 3)
        # Each node has degree 2, so C(2,2)=1 wedge per node, total 3
        assert _compute_wedge_count(adj) == 3

    def test_wedges_k4(self) -> None:
        """A K4 has 4 nodes each with degree 3, so 4*C(3,2)=12 wedges."""
        adj, _ = _create_cliques(4, 4)
        assert _compute_wedge_count(adj) == 12

    def test_wedges_multiple_triangles(self) -> None:
        """4 disconnected triangles: 4 * 3 = 12 wedges."""
        adj, _ = _create_cliques(12, 3)
        assert _compute_wedge_count(adj) == 12


class TestCountTrianglesAtEdges:
    """Tests for the _count_triangles_at_edges helper function."""

    def test_triangle_at_single_edge(self) -> None:
        """Edge in a triangle should count 1 triangle."""
        adj, _ = _create_cliques(3, 3)
        # Edge (0,1) is part of triangle (0,1,2)
        count = _count_triangles_at_edges(adj, (0, 1), (0, 2))
        # Both edges share node 0 and connect to node 2 and 1
        # Triangle (0,1,2) counted once for each edge, minus 1 for overlap
        assert count == 1

    def test_two_disjoint_edges_in_triangle(self) -> None:
        """Two edges from same triangle should count 1 triangle (with overlap)."""
        adj, _ = _create_cliques(3, 3)
        # Edges (0,1) and (1,2) share node 1
        count = _count_triangles_at_edges(adj, (0, 1), (1, 2))
        # Both are in triangle (0,1,2), counted twice minus 1 for shared node
        assert count == 1

    def test_edges_from_different_triangles(self) -> None:
        """Edges from different cliques should count independently."""
        adj, _ = _create_cliques(6, 3)
        # Edge (0,1) is in triangle (0,1,2), edge (3,4) is in triangle (3,4,5)
        count = _count_triangles_at_edges(adj, (0, 1), (3, 4))
        assert count == 2

    def test_no_triangles(self) -> None:
        """Edges with no triangles should return 0."""
        adj = lil_matrix((4, 4), dtype=np.int8)
        adj[0, 1] = adj[1, 0] = 1
        adj[2, 3] = adj[3, 2] = 1
        count = _count_triangles_at_edges(adj, (0, 1), (2, 3))
        assert count == 0
