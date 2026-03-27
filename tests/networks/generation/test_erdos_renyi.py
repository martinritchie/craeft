"""Tests for Erdős-Rényi random graph generation."""

import numpy as np
from scipy.sparse import csr_matrix

from craeft.networks.generation import random_graph


class TestRandomGraphStructure:
    """Tests for basic graph structure properties."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        matrix = random_graph(10, 0.5, rng)
        assert isinstance(matrix, csr_matrix)

    def test_shape_matches_n(self, rng: np.random.Generator) -> None:
        n = 25
        matrix = random_graph(n, 0.5, rng)
        assert matrix.shape == (n, n)

    def test_dtype_is_int8(self, rng: np.random.Generator) -> None:
        matrix = random_graph(10, 0.5, rng)
        assert matrix.dtype == np.int8

    def test_values_are_binary(self, rng: np.random.Generator) -> None:
        matrix = random_graph(50, 0.3, rng)
        unique_values = set(matrix.data)
        assert unique_values <= {0, 1}

    def test_matrix_is_symmetric(self, rng: np.random.Generator) -> None:
        matrix = random_graph(30, 0.4, rng)
        diff = matrix - matrix.T
        assert diff.nnz == 0

    def test_diagonal_is_zero(self, rng: np.random.Generator) -> None:
        matrix = random_graph(20, 0.5, rng)
        diagonal = matrix.diagonal()
        assert np.all(diagonal == 0)


class TestRandomGraphReproducibility:
    """Tests for deterministic behaviour with seeded generators."""

    def test_same_seed_same_graph(self) -> None:
        rng1 = np.random.default_rng(123)
        rng2 = np.random.default_rng(123)

        matrix1 = random_graph(50, 0.3, rng1)
        matrix2 = random_graph(50, 0.3, rng2)

        diff = matrix1 - matrix2
        assert diff.nnz == 0

    def test_different_seeds_different_graphs(self) -> None:
        rng1 = np.random.default_rng(111)
        rng2 = np.random.default_rng(222)

        matrix1 = random_graph(50, 0.3, rng1)
        matrix2 = random_graph(50, 0.3, rng2)

        diff = matrix1 - matrix2
        assert diff.nnz > 0

    def test_no_rng_produces_different_graphs(self) -> None:
        matrix1 = random_graph(50, 0.3)
        matrix2 = random_graph(50, 0.3)

        diff = matrix1 - matrix2
        assert diff.nnz > 0


class TestRandomGraphEdgeCases:
    """Tests for boundary conditions."""

    def test_p_zero_yields_empty_graph(self, rng: np.random.Generator) -> None:
        matrix = random_graph(20, 0.0, rng)
        assert matrix.nnz == 0

    def test_p_one_yields_complete_graph(self, rng: np.random.Generator) -> None:
        n = 15
        matrix = random_graph(n, 1.0, rng)
        expected_edges = n * (n - 1)  # Directed count (symmetric, so doubled)
        assert matrix.nnz == expected_edges

    def test_single_node_yields_empty_matrix(self, rng: np.random.Generator) -> None:
        matrix = random_graph(1, 0.5, rng)
        assert matrix.shape == (1, 1)
        assert matrix.nnz == 0

    def test_two_nodes_p_one_yields_single_edge(self, rng: np.random.Generator) -> None:
        matrix = random_graph(2, 1.0, rng)
        assert matrix.nnz == 2  # Symmetric: (0,1) and (1,0)


class TestRandomGraphStatistics:
    """Tests for statistical properties of G(n,p) Erdős-Rényi graphs."""

    def test_expected_edge_count(self) -> None:
        """Mean edges over ensemble should approximate n(n-1)/2 * p."""
        n, p = 100, 0.15
        runs = 50

        edge_counts = [
            random_graph(n, p, np.random.default_rng(seed)).nnz // 2
            for seed in range(runs)
        ]

        mean_edges = np.mean(edge_counts)
        expected = n * (n - 1) / 2 * p

        assert abs(mean_edges - expected) < 0.15 * expected

    def test_expected_mean_degree(self) -> None:
        """Mean degree over ensemble should approximate (n-1) * p."""
        n, p = 100, 0.2
        runs = 50

        mean_degrees = []
        for seed in range(runs):
            matrix = random_graph(n, p, np.random.default_rng(seed))
            degrees = np.array(matrix.sum(axis=1)).flatten()
            mean_degrees.append(degrees.mean())

        ensemble_mean = np.mean(mean_degrees)
        expected = (n - 1) * p

        assert abs(ensemble_mean - expected) < 0.15 * expected

    def test_degree_variance(self) -> None:
        """Degree variance should approximate (n-1) * p * (1-p)."""
        n, p = 100, 0.3
        runs = 50

        variances = []
        for seed in range(runs):
            matrix = random_graph(n, p, np.random.default_rng(seed))
            degrees = np.array(matrix.sum(axis=1)).flatten()
            variances.append(degrees.var())

        mean_variance = np.mean(variances)
        expected = (n - 1) * p * (1 - p)

        assert abs(mean_variance - expected) < 0.2 * expected

    def test_degree_distribution_symmetric(self) -> None:
        """Degree distribution should be roughly symmetric around mean."""
        n, p = 200, 0.5
        rng = np.random.default_rng(42)

        matrix = random_graph(n, p, rng)
        degrees = np.array(matrix.sum(axis=1)).flatten()

        mean_degree = degrees.mean()
        median_degree = np.median(degrees)

        # For binomial with p=0.5, mean ≈ median
        assert abs(mean_degree - median_degree) < 5

    def test_no_isolated_nodes_at_high_p(self) -> None:
        """At high p, all nodes should have at least one edge."""
        n, p = 50, 0.5
        rng = np.random.default_rng(123)

        matrix = random_graph(n, p, rng)
        degrees = np.array(matrix.sum(axis=1)).flatten()
        isolated = np.sum(degrees == 0)

        assert isolated == 0

    def test_sparse_graph_has_isolated_nodes(self) -> None:
        """At very low p, some nodes should be isolated."""
        n, p = 100, 0.01
        runs = 20

        total_isolated = 0
        for seed in range(runs):
            matrix = random_graph(n, p, np.random.default_rng(seed))
            degrees = np.array(matrix.sum(axis=1)).flatten()
            total_isolated += np.sum(degrees == 0)

        # Expect some isolated nodes across runs
        assert total_isolated > 0
