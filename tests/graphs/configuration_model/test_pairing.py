"""Tests for single stub pairing via Connector."""

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.graphs.configuration_model.connection import Connector


def _pair(degrees: NDArray[np.int_], rng: np.random.Generator) -> csr_matrix:
    """Helper: connect singles and return adjacency."""
    connector = Connector(len(degrees), rng)
    connector.connect_singles(degrees)
    return connector.to_csr()


class TestConnectorSinglesValidity:
    """Structural properties of paired graphs."""

    def test_adjacency_is_symmetric(self) -> None:
        adj = _pair(np.array([3, 3, 2, 2]), np.random.default_rng(42))
        assert (adj - adj.T).nnz == 0

    def test_no_self_loops(self) -> None:
        adj = _pair(np.array([4, 4, 4, 4]), np.random.default_rng(42))
        assert np.all(adj.diagonal() == 0)

    def test_no_multi_edges(self) -> None:
        adj = _pair(np.array([4, 4, 4, 4]), np.random.default_rng(42))
        assert adj.data.max() <= 1

    def test_shape_matches_n(self) -> None:
        adj = _pair(np.array([2, 2, 2, 2, 2]), np.random.default_rng(42))
        assert adj.shape == (5, 5)

    def test_values_are_binary(self) -> None:
        adj = _pair(np.array([3, 3, 2, 2]), np.random.default_rng(42))
        assert set(adj.data) <= {0, 1}


class TestConnectorSinglesEdgeCases:
    def test_all_zero_degrees(self) -> None:
        adj = _pair(np.array([0, 0, 0]), np.random.default_rng(42))
        assert adj.nnz == 0

    def test_two_nodes_degree_one(self) -> None:
        adj = _pair(np.array([1, 1]), np.random.default_rng(42))
        assert adj.nnz == 2

    def test_odd_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            _pair(np.array([1, 2]), np.random.default_rng(42))


class TestConnectorSinglesReproducibility:
    def test_same_seed_same_graph(self) -> None:
        degrees = np.array([3, 3, 3, 3, 2, 2])
        a1 = _pair(degrees, np.random.default_rng(42))
        a2 = _pair(degrees, np.random.default_rng(42))
        assert (a1 != a2).nnz == 0

    def test_different_seeds_can_differ(self) -> None:
        degrees = np.full(50, 4)
        a1 = _pair(degrees, np.random.default_rng(1))
        a2 = _pair(degrees, np.random.default_rng(2))
        assert (a1 != a2).nnz > 0


class TestConnectorSinglesDegreePreservation:
    """Realised degrees should approximate input (lost edges from cleanup)."""

    @pytest.mark.parametrize("seed", range(10))
    def test_realised_degrees_at_most_input(self, seed: int) -> None:
        degrees = np.full(100, 6)
        adj = _pair(degrees, np.random.default_rng(seed))
        realised = np.asarray(adj.sum(axis=1)).flatten()
        assert np.all(realised <= degrees)

    def test_mean_degree_close_to_input(self) -> None:
        """For sparse graphs, cleanup removes few edges."""
        target = 4
        degrees = np.full(500, target)
        adj = _pair(degrees, np.random.default_rng(42))
        realised_mean = np.asarray(adj.sum(axis=1)).flatten().mean()
        assert abs(realised_mean - target) < 0.5
