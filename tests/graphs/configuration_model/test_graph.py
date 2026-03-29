"""Tests for ConfigModelGraph."""

import numpy as np
import pytest
from scipy.stats import poisson

from craeft.graphs.base import UndirectedGraph
from craeft.graphs.configuration_model import (
    ConfigModelConfig,
    ConfigModelGraph,
)
from craeft.graphs.configuration_model.sequence import sample_degree_sequence

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigModelConfig:
    def test_valid_config(self) -> None:
        config = ConfigModelConfig(n=4, degrees=np.array([2, 2, 2, 2]))
        assert config.n == 4

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            ConfigModelConfig(n=3, degrees=np.array([2, 2, 2, 2]))

    def test_negative_degree_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ConfigModelConfig(n=3, degrees=np.array([2, -1, 2]))

    def test_odd_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            ConfigModelConfig(n=3, degrees=np.array([1, 1, 1]))


# ---------------------------------------------------------------------------
# Construction and type
# ---------------------------------------------------------------------------


class TestConfigModelGraphConstruction:
    def test_is_undirected_graph(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=4, degrees=np.array([2, 2, 2, 2]))
        graph = ConfigModelGraph.from_config(config, rng)
        assert isinstance(graph, UndirectedGraph)

    def test_n_nodes_matches_config(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=50, degrees=np.full(50, 4))
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.n_nodes == 50

    def test_adjacency_is_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=30, degrees=np.full(30, 6))
        csr = ConfigModelGraph.from_config(config, rng).to_csr()
        assert (csr - csr.T).nnz == 0

    def test_no_self_loops(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=20, degrees=np.full(20, 4))
        csr = ConfigModelGraph.from_config(config, rng).to_csr()
        assert np.all(csr.diagonal() == 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestConfigModelGraphEdgeCases:
    def test_all_zero_degrees(self) -> None:
        rng = np.random.default_rng(42)
        graph = ConfigModelGraph.from_config(
            ConfigModelConfig(n=10, degrees=np.zeros(10, dtype=np.int_)), rng
        )
        assert graph.n_edges == 0

    def test_two_nodes_degree_one(self) -> None:
        rng = np.random.default_rng(42)
        graph = ConfigModelGraph.from_config(
            ConfigModelConfig(n=2, degrees=np.array([1, 1])), rng
        )
        assert graph.n_edges == 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestConfigModelGraphReproducibility:
    def test_same_seed_same_graph(self) -> None:
        degrees = np.full(50, 4)
        config = ConfigModelConfig(n=50, degrees=degrees)
        g1 = ConfigModelGraph.from_config(config, np.random.default_rng(42))
        g2 = ConfigModelGraph.from_config(config, np.random.default_rng(42))
        assert g1 == g2

    def test_different_seeds_different_graphs(self) -> None:
        degrees = np.full(50, 4)
        config = ConfigModelConfig(n=50, degrees=degrees)
        g1 = ConfigModelGraph.from_config(config, np.random.default_rng(1))
        g2 = ConfigModelGraph.from_config(config, np.random.default_rng(2))
        assert g1 != g2


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestConfigModelGraphProperties:
    def test_clustering_coefficient_type(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=50, degrees=np.full(50, 6))
        graph = ConfigModelGraph.from_config(config, rng)
        assert isinstance(graph.clustering_coefficient, float)

    def test_clustering_near_zero_for_sparse_graph(self) -> None:
        """Vanilla CM produces near-zero clustering."""
        rng = np.random.default_rng(42)
        degrees = sample_degree_sequence(500, poisson(5), rng)
        config = ConfigModelConfig(n=500, degrees=degrees)
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.clustering_coefficient < 0.1


# ---------------------------------------------------------------------------
# Integration: full pipeline from distribution to graph
# ---------------------------------------------------------------------------


class TestConfigModelPipeline:
    def test_sample_then_generate(self) -> None:
        rng = np.random.default_rng(42)
        degrees = sample_degree_sequence(200, poisson(4), rng)
        config = ConfigModelConfig(n=200, degrees=degrees)
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.n_nodes == 200
        assert graph.n_edges > 0

    def test_mean_degree_approximates_input(self) -> None:
        rng = np.random.default_rng(42)
        target = 6
        degrees = sample_degree_sequence(500, poisson(target), rng)
        config = ConfigModelConfig(n=500, degrees=degrees)
        graph = ConfigModelGraph.from_config(config, rng)
        assert abs(graph.degrees.mean() - target) < 1.0
