"""Integration tests for the CMA entry point."""

import numpy as np
import pytest

from craeft.networks.generation.configuration_model.cma import cma
from craeft.networks.generation.configuration_model.subgraph_spec import SubgraphSpec
from craeft.networks.generation.motifs import G2, G7, G8
from craeft.networks.metrics import global_clustering_coefficient


class TestCMAStructure:
    """Tests for basic output structure."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        degrees = np.full(100, 5)
        spec = SubgraphSpec(motif=G2, sequence=np.full(100, 1))
        adj = cma(degrees, [spec], rng=rng)

        assert adj.shape == (100, 100)
        assert (adj != adj.T).nnz == 0  # symmetric

    def test_empty_specs_falls_back_to_simple_cm(
        self,
        rng: np.random.Generator,
    ) -> None:
        degrees = np.full(50, 4)
        adj = cma(degrees, [], rng=rng)
        assert adj.shape == (50, 50)

    def test_no_self_loops(self, rng: np.random.Generator) -> None:
        degrees = np.full(100, 6)
        spec = SubgraphSpec(motif=G2, sequence=np.full(100, 2))
        adj = cma(degrees, [spec], rng=rng)

        assert adj.diagonal().sum() == 0


class TestCMAClustering:
    """Tests for clustering coefficient behavior."""

    def test_triangles_produce_clustering(self, rng: np.random.Generator) -> None:
        degrees = np.full(300, 5)
        spec = SubgraphSpec(motif=G2, sequence=np.full(300, 2))
        adj = cma(degrees, [spec], rng=rng)

        cc = global_clustering_coefficient(adj)
        assert cc > 0.05

    def test_k4_produces_higher_clustering(self, rng: np.random.Generator) -> None:
        degrees = np.full(300, 6)
        spec_g2 = SubgraphSpec(motif=G2, sequence=np.full(300, 1))
        spec_g8 = SubgraphSpec(motif=G8, sequence=np.full(300, 1))

        adj_g2 = cma(degrees, [spec_g2], rng=np.random.default_rng(42))
        adj_g8 = cma(degrees, [spec_g8], rng=np.random.default_rng(42))

        cc_g2 = global_clustering_coefficient(adj_g2)
        cc_g8 = global_clustering_coefficient(adj_g8)

        assert cc_g8 > cc_g2

    def test_empty_specs_low_clustering(self, rng: np.random.Generator) -> None:
        degrees = np.full(300, 5)
        adj = cma(degrees, [], rng=rng)

        cc = global_clustering_coefficient(adj)
        assert cc < 0.05


class TestCMAIncompleteMotifs:
    """Tests for incomplete (mixed-cardinality) motifs."""

    def test_g7_diamond_produces_network(self, rng: np.random.Generator) -> None:
        """G7 is incomplete but should still produce a valid network."""
        degrees = np.full(200, 8)
        spec = SubgraphSpec(motif=G7, sequence=np.full(200, 2))
        adj = cma(degrees, [spec], rng=rng)

        assert adj.shape == (200, 200)
        assert adj.diagonal().sum() == 0


class TestCMAReproducibility:
    """Tests for deterministic output."""

    def test_same_seed_same_result(self) -> None:
        degrees = np.full(100, 5)
        spec = SubgraphSpec(motif=G2, sequence=np.full(100, 1))

        adj1 = cma(degrees, [spec], rng=np.random.default_rng(99))
        adj2 = cma(degrees, [spec], rng=np.random.default_rng(99))

        assert (adj1 != adj2).nnz == 0

    def test_different_seeds_different_result(self) -> None:
        degrees = np.full(100, 5)
        spec = SubgraphSpec(motif=G2, sequence=np.full(100, 1))

        adj1 = cma(degrees, [spec], rng=np.random.default_rng(1))
        adj2 = cma(degrees, [spec], rng=np.random.default_rng(2))

        assert (adj1 != adj2).nnz > 0


class TestCMAValidation:
    """Tests for input validation."""

    def test_rejects_mismatched_sequence_length(
        self,
        rng: np.random.Generator,
    ) -> None:
        degrees = np.full(100, 5)
        spec = SubgraphSpec(motif=G2, sequence=np.full(50, 1))  # wrong length

        with pytest.raises(ValueError, match="length"):
            cma(degrees, [spec], rng=rng)
