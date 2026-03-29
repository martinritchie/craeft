"""Tests for ErdosRenyiGraph."""

import numpy as np
import pytest
from scipy.stats import binom, ks_1samp

from craeft.graphs.base import UndirectedGraph
from craeft.graphs.erdos_renyi import ErdosRenyiConfig, ErdosRenyiGraph

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestErdosRenyiConfig:
    def test_valid_config(self) -> None:
        config = ErdosRenyiConfig(n=10, p=0.5)
        assert config.n == 10
        assert config.p == 0.5

    @pytest.mark.parametrize("p", [-0.1, 1.1, 2.0])
    def test_invalid_p_raises(self, p: float) -> None:
        with pytest.raises(ValueError, match="p must be in"):
            ErdosRenyiConfig(n=10, p=p)

    def test_boundary_p_zero(self) -> None:
        ErdosRenyiConfig(n=10, p=0.0)

    def test_boundary_p_one(self) -> None:
        ErdosRenyiConfig(n=10, p=1.0)


# ---------------------------------------------------------------------------
# Construction and type
# ---------------------------------------------------------------------------


class TestErdosRenyiGraphConstruction:
    def test_is_undirected_graph(self) -> None:
        rng = np.random.default_rng(42)
        config = ErdosRenyiConfig(n=10, p=0.5)
        graph = ErdosRenyiGraph.from_config(config, rng)
        assert isinstance(graph, UndirectedGraph)

    def test_n_nodes_matches_config(self) -> None:
        rng = np.random.default_rng(42)
        config = ErdosRenyiConfig(n=50, p=0.3)
        graph = ErdosRenyiGraph.from_config(config, rng)
        assert graph.n_nodes == 50

    def test_adjacency_is_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        config = ErdosRenyiConfig(n=30, p=0.4)
        csr = ErdosRenyiGraph.from_config(config, rng).to_csr()
        assert (csr - csr.T).nnz == 0

    def test_no_self_loops(self) -> None:
        rng = np.random.default_rng(42)
        config = ErdosRenyiConfig(n=20, p=0.5)
        csr = ErdosRenyiGraph.from_config(config, rng).to_csr()
        assert np.all(csr.diagonal() == 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestErdosRenyiGraphEdgeCases:
    def test_p_zero_yields_no_edges(self) -> None:
        rng = np.random.default_rng(42)
        graph = ErdosRenyiGraph.from_config(ErdosRenyiConfig(n=20, p=0.0), rng)
        assert graph.n_edges == 0

    def test_p_one_yields_complete_graph(self) -> None:
        rng = np.random.default_rng(42)
        n = 15
        graph = ErdosRenyiGraph.from_config(ErdosRenyiConfig(n=n, p=1.0), rng)
        assert graph.n_edges == n * (n - 1) // 2

    def test_single_node(self) -> None:
        rng = np.random.default_rng(42)
        graph = ErdosRenyiGraph.from_config(ErdosRenyiConfig(n=1, p=0.5), rng)
        assert graph.n_nodes == 1
        assert graph.n_edges == 0


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestErdosRenyiGraphReproducibility:
    def test_same_seed_same_graph(self) -> None:
        config = ErdosRenyiConfig(n=50, p=0.3)
        g1 = ErdosRenyiGraph.from_config(config, np.random.default_rng(123))
        g2 = ErdosRenyiGraph.from_config(config, np.random.default_rng(123))
        assert g1 == g2

    def test_different_seeds_different_graphs(self) -> None:
        config = ErdosRenyiConfig(n=50, p=0.3)
        g1 = ErdosRenyiGraph.from_config(config, np.random.default_rng(111))
        g2 = ErdosRenyiGraph.from_config(config, np.random.default_rng(222))
        assert g1 != g2


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestErdosRenyiGraphProperties:
    def test_clustering_coefficient_type(self) -> None:
        rng = np.random.default_rng(42)
        graph = ErdosRenyiGraph.from_config(ErdosRenyiConfig(n=50, p=0.3), rng)
        assert isinstance(graph.clustering_coefficient, float)

    def test_clustering_zero_for_empty_graph(self) -> None:
        rng = np.random.default_rng(42)
        graph = ErdosRenyiGraph.from_config(ErdosRenyiConfig(n=20, p=0.0), rng)
        assert graph.clustering_coefficient == 0.0

    def test_clustering_one_for_complete_small_graph(self) -> None:
        rng = np.random.default_rng(42)
        graph = ErdosRenyiGraph.from_config(ErdosRenyiConfig(n=5, p=1.0), rng)
        assert graph.clustering_coefficient == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Statistical properties
# ---------------------------------------------------------------------------


class TestErdosRenyiGraphStatistics:
    def test_expected_edge_count(self) -> None:
        """Mean edges over ensemble approximates n(n-1)/2 * p."""
        n, p = 100, 0.15
        config = ErdosRenyiConfig(n=n, p=p)
        edge_counts = [
            ErdosRenyiGraph.from_config(config, np.random.default_rng(seed)).n_edges
            for seed in range(50)
        ]
        mean_edges = np.mean(edge_counts)
        expected = n * (n - 1) / 2 * p
        assert abs(mean_edges - expected) < 0.15 * expected

    def test_expected_mean_degree(self) -> None:
        """Mean degree over ensemble approximates (n-1) * p."""
        n, p = 100, 0.2
        config = ErdosRenyiConfig(n=n, p=p)
        mean_degrees = [
            ErdosRenyiGraph.from_config(
                config, np.random.default_rng(seed)
            ).degrees.mean()
            for seed in range(50)
        ]
        ensemble_mean = np.mean(mean_degrees)
        expected = (n - 1) * p
        assert abs(ensemble_mean - expected) < 0.15 * expected

    def test_edge_count_is_binomial(self) -> None:
        """Edge counts follow Binomial(max_edges, p)."""
        n, p = 30, 0.3
        max_edges = n * (n - 1) // 2
        config = ErdosRenyiConfig(n=n, p=p)
        runs = 500

        edge_counts = np.array(
            [
                ErdosRenyiGraph.from_config(config, np.random.default_rng(seed)).n_edges
                for seed in range(runs)
            ]
        )

        # KS test against Binomial(max_edges, p)
        # Conservative for discrete distributions (won't falsely reject)
        _, p_value = ks_1samp(edge_counts, binom.cdf, args=(max_edges, p))
        assert p_value > 0.01
