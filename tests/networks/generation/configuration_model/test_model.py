"""Tests for Clustered Configuration Model (CCM) network generation."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix
from scipy.stats import poisson

from craeft.networks.generation.configuration_model import (
    configuration_model,
    sample_network,
)
from craeft.networks.metrics import global_clustering_coefficient


class TestCCMStructure:
    """Tests for basic graph structure properties."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        degrees = np.array([4, 4, 4, 4])
        matrix = configuration_model(degrees, phi=0.3, rng=rng)
        assert isinstance(matrix, csr_matrix)

    def test_shape_matches_n(self, rng: np.random.Generator) -> None:
        degrees = np.array([4, 4, 4, 4, 4, 4])
        matrix = configuration_model(degrees, phi=0.3, rng=rng)
        assert matrix.shape == (6, 6)

    def test_matrix_is_symmetric(self, rng: np.random.Generator) -> None:
        degrees = np.array([4, 4, 4, 4, 4, 4])
        matrix = configuration_model(degrees, phi=0.5, rng=rng)
        diff = matrix - matrix.T
        assert diff.nnz == 0

    def test_no_self_loops_by_default(self, rng: np.random.Generator) -> None:
        degrees = np.array([4, 4, 4, 4])
        matrix = configuration_model(degrees, phi=0.5, rng=rng)
        assert np.all(matrix.diagonal() == 0)


class TestCCMClustering:
    """Tests for clustering coefficient behavior."""

    def test_phi_zero_produces_low_clustering(self) -> None:
        """With phi=0, no triangles are intentionally formed."""
        degrees = np.full(200, 6)
        rng = np.random.default_rng(42)
        matrix = configuration_model(degrees, phi=0.0, rng=rng)
        clustering = global_clustering_coefficient(matrix)
        # Should be low (random chance of triangles in config model)
        assert clustering < 0.1

    def test_higher_phi_produces_higher_clustering(self) -> None:
        """Clustering should increase monotonically with phi."""
        degrees = np.full(300, 6)
        phi_values = [0.0, 0.3, 0.6, 0.9]
        clusterings = []

        for phi in phi_values:
            rng = np.random.default_rng(42)
            matrix = configuration_model(degrees, phi=phi, rng=rng)
            clusterings.append(global_clustering_coefficient(matrix))

        # Each subsequent phi should produce higher clustering
        for i in range(len(clusterings) - 1):
            phi_prev, phi_next = phi_values[i], phi_values[i + 1]
            c_prev, c_next = clusterings[i], clusterings[i + 1]
            assert c_next > c_prev, (
                f"Clustering should increase: phi={phi_prev}->{phi_next}, "
                f"clustering={c_prev:.3f}->{c_next:.3f}"
            )

    def test_phi_one_produces_high_clustering(self) -> None:
        """With phi=1, maximum triangles are formed."""
        degrees = np.full(200, 6)
        rng = np.random.default_rng(42)
        matrix = configuration_model(degrees, phi=1.0, rng=rng)
        clustering = global_clustering_coefficient(matrix)
        # Should be relatively high
        assert clustering > 0.2


class TestCCMEdgeCases:
    """Tests for boundary conditions."""

    def test_odd_degree_sum_raises(self, rng: np.random.Generator) -> None:
        degrees = np.array([3, 3, 3])  # Sum = 9 (odd)
        with pytest.raises(ValueError, match="even"):
            configuration_model(degrees, phi=0.5, rng=rng)

    def test_phi_out_of_range_raises(self, rng: np.random.Generator) -> None:
        degrees = np.array([4, 4, 4, 4])
        with pytest.raises(ValueError, match="phi"):
            configuration_model(degrees, phi=1.5, rng=rng)
        with pytest.raises(ValueError, match="phi"):
            configuration_model(degrees, phi=-0.1, rng=rng)

    def test_zero_degrees_yields_empty_graph(self, rng: np.random.Generator) -> None:
        degrees = np.array([0, 0, 0, 0])
        matrix = configuration_model(degrees, phi=0.5, rng=rng)
        assert matrix.nnz == 0

    def test_small_degrees_work(self, rng: np.random.Generator) -> None:
        """Nodes with degree 1 can't form triangles but shouldn't crash."""
        degrees = np.array([1, 1, 2, 2])
        matrix = configuration_model(degrees, phi=0.5, rng=rng)
        assert matrix.shape == (4, 4)


class TestCCMReproducibility:
    """Tests for deterministic behavior with seeded generators."""

    def test_same_seed_same_graph(self) -> None:
        degrees = np.array([4, 4, 4, 4, 4, 4])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        matrix1 = configuration_model(degrees, phi=0.5, rng=rng1)
        matrix2 = configuration_model(degrees, phi=0.5, rng=rng2)

        diff = matrix1 - matrix2
        assert diff.nnz == 0

    def test_different_seeds_different_graphs(self) -> None:
        degrees = np.array([4, 4, 4, 4, 4, 4])
        rng1 = np.random.default_rng(111)
        rng2 = np.random.default_rng(222)

        matrix1 = configuration_model(degrees, phi=0.5, rng=rng1)
        matrix2 = configuration_model(degrees, phi=0.5, rng=rng2)

        diff = matrix1 - matrix2
        assert diff.nnz > 0


class TestSampleClusteredNetwork:
    """Tests for the convenience sampling function."""

    def test_returns_correct_size(self) -> None:
        rng = np.random.default_rng(42)
        matrix = sample_network(
            n=100, pmf=poisson(4).pmf, max_degree=20, phi=0.3, rng=rng
        )
        assert matrix.shape == (100, 100)

    def test_produces_clustering(self) -> None:
        rng = np.random.default_rng(42)
        matrix = sample_network(
            n=200, pmf=poisson(6).pmf, max_degree=30, phi=0.6, rng=rng
        )
        clustering = global_clustering_coefficient(matrix)
        assert clustering > 0.05  # Should have some clustering
