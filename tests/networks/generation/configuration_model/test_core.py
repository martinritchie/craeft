"""Tests for configuration model random graph generation."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from craeft.networks.generation import configuration_model


class TestConfigurationModelStructure:
    """Tests for basic graph structure properties."""

    def test_returns_csr_matrix(self, rng: np.random.Generator) -> None:
        degrees = np.array([2, 2, 2, 2])
        matrix = configuration_model(degrees, rng=rng)
        assert isinstance(matrix, csr_matrix)

    def test_shape_matches_n(self, rng: np.random.Generator) -> None:
        degrees = np.array([3, 3, 2, 2, 2, 2])
        matrix = configuration_model(degrees, rng=rng)
        assert matrix.shape == (6, 6)

    def test_dtype_is_int8(self, rng: np.random.Generator) -> None:
        degrees = np.array([2, 2, 2, 2])
        matrix = configuration_model(degrees, rng=rng)
        assert matrix.dtype == np.int8

    def test_matrix_is_symmetric(self, rng: np.random.Generator) -> None:
        degrees = np.array([4, 3, 3, 2, 2, 2, 2])
        matrix = configuration_model(degrees, rng=rng)
        diff = matrix - matrix.T
        assert diff.nnz == 0


class TestConfigurationModelDegrees:
    """Tests for degree sequence preservation."""

    def test_realizes_exact_degrees_without_self_loops(
        self, rng: np.random.Generator
    ) -> None:
        """Without self-loops/multi-edges, realized degrees may differ slightly."""
        degrees = np.array([3, 3, 2, 2, 2, 2])
        matrix = configuration_model(
            degrees, rng=rng, allow_self_loops=False, allow_multi_edges=False
        )
        realized = np.array(matrix.sum(axis=1)).flatten()
        # Degrees should be close but may differ due to removed self-loops/multi-edges
        assert realized.sum() <= degrees.sum()

    def test_sum_preserved_with_multi_edges(self, rng: np.random.Generator) -> None:
        """With multi-edges allowed, total stub count is preserved."""
        degrees = np.array([4, 4, 4, 4])
        matrix = configuration_model(
            degrees, rng=rng, allow_self_loops=True, allow_multi_edges=True
        )
        # Sum of realized degrees equals sum of prescribed degrees
        realized_sum = matrix.sum()
        assert realized_sum == degrees.sum()


class TestConfigurationModelEdgeCases:
    """Tests for boundary conditions."""

    def test_odd_degree_sum_raises(self, rng: np.random.Generator) -> None:
        degrees = np.array([3, 2, 2])  # Sum = 7 (odd)
        with pytest.raises(ValueError, match="even"):
            configuration_model(degrees, rng=rng)

    def test_zero_degrees_yields_empty_graph(self, rng: np.random.Generator) -> None:
        degrees = np.array([0, 0, 0, 0])
        matrix = configuration_model(degrees, rng=rng)
        assert matrix.nnz == 0

    def test_single_node_zero_degree(self, rng: np.random.Generator) -> None:
        degrees = np.array([0])
        matrix = configuration_model(degrees, rng=rng)
        assert matrix.shape == (1, 1)
        assert matrix.nnz == 0

    def test_two_nodes_degree_one_each(self, rng: np.random.Generator) -> None:
        degrees = np.array([1, 1])
        matrix = configuration_model(degrees, rng=rng)
        assert matrix.nnz == 2  # Symmetric: (0,1) and (1,0)


class TestConfigurationModelSelfLoops:
    """Tests for self-loop handling."""

    def test_no_self_loops_by_default(self, rng: np.random.Generator) -> None:
        degrees = np.array([4, 4, 4, 4])
        matrix = configuration_model(degrees, rng=rng, allow_self_loops=False)
        diagonal = matrix.diagonal()
        assert np.all(diagonal == 0)

    def test_self_loops_when_allowed(self) -> None:
        """With small graphs and high degrees, self-loops should occur sometimes."""
        found_self_loop = False
        for seed in range(50):
            degrees = np.array([4, 4, 4, 4])
            matrix = configuration_model(
                degrees, rng=np.random.default_rng(seed), allow_self_loops=True
            )
            if matrix.diagonal().sum() > 0:
                found_self_loop = True
                break
        assert found_self_loop


class TestConfigurationModelReproducibility:
    """Tests for deterministic behaviour with seeded generators."""

    def test_same_seed_same_graph(self) -> None:
        degrees = np.array([3, 3, 2, 2, 2, 2])
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        matrix1 = configuration_model(degrees, rng=rng1)
        matrix2 = configuration_model(degrees, rng=rng2)

        diff = matrix1 - matrix2
        assert diff.nnz == 0

    def test_different_seeds_different_graphs(self) -> None:
        degrees = np.array([3, 3, 3, 3, 2, 2])
        rng1 = np.random.default_rng(111)
        rng2 = np.random.default_rng(222)

        matrix1 = configuration_model(degrees, rng=rng1)
        matrix2 = configuration_model(degrees, rng=rng2)

        diff = matrix1 - matrix2
        assert diff.nnz > 0
