"""Tests for network generator protocol implementations."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from craeft.networks.generation.generator import (
    BigVRewiringGenerator,
    ConfigurationModelGenerator,
    ErdosRenyiGenerator,
    MotifDecompositionGenerator,
    PoissonNetworkGenerator,
)


class TestErdosRenyiGenerator:
    """Tests for ErdosRenyiGenerator."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        gen = ErdosRenyiGenerator(n=50, p=0.1)
        result = gen.generate(rng)
        assert isinstance(result, csr_matrix)

    def test_shape_matches_n(self, rng: np.random.Generator) -> None:
        gen = ErdosRenyiGenerator(n=30, p=0.2)
        result = gen.generate(rng)
        assert result.shape == (30, 30)

    def test_symmetric(self, rng: np.random.Generator) -> None:
        gen = ErdosRenyiGenerator(n=50, p=0.3)
        result = gen.generate(rng)
        assert (result - result.T).nnz == 0

    def test_reproducible_with_same_seed(self) -> None:
        gen = ErdosRenyiGenerator(n=50, p=0.2)
        a = gen.generate(np.random.default_rng(99))
        b = gen.generate(np.random.default_rng(99))
        assert (a - b).nnz == 0

    def test_different_seeds_differ(self) -> None:
        gen = ErdosRenyiGenerator(n=50, p=0.2)
        a = gen.generate(np.random.default_rng(1))
        b = gen.generate(np.random.default_rng(2))
        assert (a - b).nnz > 0


class TestConfigurationModelGenerator:
    """Tests for ConfigurationModelGenerator."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        degrees = np.array([2, 2, 2, 2], dtype=np.int_)
        gen = ConfigurationModelGenerator(degrees=degrees)
        result = gen.generate(rng)
        assert isinstance(result, csr_matrix)

    def test_shape_matches_degree_sequence(self, rng: np.random.Generator) -> None:
        degrees = np.array([3, 3, 3, 3, 3, 3], dtype=np.int_)
        gen = ConfigurationModelGenerator(degrees=degrees)
        result = gen.generate(rng)
        assert result.shape == (6, 6)

    def test_phi_zero_default(self) -> None:
        gen = ConfigurationModelGenerator(degrees=np.array([2, 2], dtype=np.int_))
        assert gen.phi == 0.0

    def test_symmetric(self, rng: np.random.Generator) -> None:
        degrees = np.full(20, 4, dtype=np.int_)
        gen = ConfigurationModelGenerator(degrees=degrees)
        result = gen.generate(rng)
        assert (result - result.T).nnz == 0


class TestPoissonNetworkGenerator:
    """Tests for PoissonNetworkGenerator."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        gen = PoissonNetworkGenerator(n=100, mean_degree=5.0, max_degree=20)
        result = gen.generate(rng)
        assert isinstance(result, csr_matrix)

    def test_shape_matches_n(self, rng: np.random.Generator) -> None:
        gen = PoissonNetworkGenerator(n=80, mean_degree=4.0, max_degree=15)
        result = gen.generate(rng)
        assert result.shape == (80, 80)

    def test_mean_degree_approximately_correct(self) -> None:
        gen = PoissonNetworkGenerator(n=500, mean_degree=6.0, max_degree=25)
        result = gen.generate(np.random.default_rng(42))
        actual_mean = np.array(result.sum(axis=1)).flatten().mean()
        assert abs(actual_mean - 6.0) < 1.5

    def test_phi_zero_default(self) -> None:
        gen = PoissonNetworkGenerator(n=100, mean_degree=5.0, max_degree=20)
        assert gen.phi == 0.0

    def test_reproducible_with_same_seed(self) -> None:
        gen = PoissonNetworkGenerator(n=100, mean_degree=5.0, max_degree=20)
        a = gen.generate(np.random.default_rng(42))
        b = gen.generate(np.random.default_rng(42))
        assert (a - b).nnz == 0


class TestBigVRewiringGenerator:
    """Tests for BigVRewiringGenerator."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        base = ConfigurationModelGenerator(degrees=np.full(100, 4, dtype=np.int_))
        gen = BigVRewiringGenerator(base=base, target_clustering=0.1)
        result = gen.generate(rng)
        assert isinstance(result, csr_matrix)

    def test_shape_preserved(self, rng: np.random.Generator) -> None:
        base = ConfigurationModelGenerator(degrees=np.full(50, 4, dtype=np.int_))
        gen = BigVRewiringGenerator(base=base, target_clustering=0.1)
        result = gen.generate(rng)
        assert result.shape == (50, 50)

    def test_symmetric(self, rng: np.random.Generator) -> None:
        base = ConfigurationModelGenerator(degrees=np.full(100, 4, dtype=np.int_))
        gen = BigVRewiringGenerator(base=base, target_clustering=0.1)
        result = gen.generate(rng)
        assert (result - result.T).nnz == 0


class TestMotifDecompositionGenerator:
    """Tests for MotifDecompositionGenerator."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        gen = MotifDecompositionGenerator(
            num_nodes=12, clique_size=3, target_clustering=0.3
        )
        result = gen.generate(rng)
        assert isinstance(result, csr_matrix)

    def test_shape_matches_num_nodes(self, rng: np.random.Generator) -> None:
        gen = MotifDecompositionGenerator(
            num_nodes=12, clique_size=3, target_clustering=0.3
        )
        result = gen.generate(rng)
        assert result.shape == (12, 12)

    def test_symmetric(self, rng: np.random.Generator) -> None:
        gen = MotifDecompositionGenerator(
            num_nodes=12, clique_size=3, target_clustering=0.3
        )
        result = gen.generate(rng)
        assert (result - result.T).nnz == 0


class TestGeneratorPicklability:
    """Verify generators can be pickled (required for multiprocessing)."""

    @pytest.mark.parametrize(
        "generator",
        [
            ErdosRenyiGenerator(n=10, p=0.5),
            ConfigurationModelGenerator(degrees=np.array([2, 2, 2, 2], dtype=np.int_)),
            PoissonNetworkGenerator(n=10, mean_degree=3.0, max_degree=10),
            MotifDecompositionGenerator(
                num_nodes=6, clique_size=3, target_clustering=0.3
            ),
        ],
        ids=["erdos_renyi", "config_model", "poisson", "motif_decomp"],
    )
    def test_pickle_roundtrip(self, generator) -> None:
        import pickle

        restored = pickle.loads(pickle.dumps(generator))
        rng = np.random.default_rng(42)
        original = generator.generate(np.random.default_rng(42))
        result = restored.generate(rng)
        assert (original - result).nnz == 0
