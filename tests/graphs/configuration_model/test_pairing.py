"""Tests for stub pairing algorithm."""

import numpy as np
import pytest

from craeft.graphs.configuration_model.connection import connect_singles


class TestPairStubsValidity:
    """Structural properties of paired graphs."""

    def test_adjacency_is_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        degrees = np.array([3, 3, 2, 2])
        adj = connect_singles(degrees, rng)
        assert (adj - adj.T).nnz == 0

    def test_no_self_loops(self) -> None:
        rng = np.random.default_rng(42)
        degrees = np.array([4, 4, 4, 4])
        adj = connect_singles(degrees, rng)
        assert np.all(adj.diagonal() == 0)

    def test_no_multi_edges(self) -> None:
        rng = np.random.default_rng(42)
        degrees = np.array([4, 4, 4, 4])
        adj = connect_singles(degrees, rng)
        assert adj.data.max() <= 1

    def test_shape_matches_n(self) -> None:
        rng = np.random.default_rng(42)
        degrees = np.array([2, 2, 2, 2, 2])
        adj = connect_singles(degrees, rng)
        assert adj.shape == (5, 5)

    def test_values_are_binary(self) -> None:
        rng = np.random.default_rng(42)
        degrees = np.array([3, 3, 2, 2])
        adj = connect_singles(degrees, rng)
        assert set(adj.data) <= {0, 1}


class TestPairStubsEdgeCases:
    def test_all_zero_degrees(self) -> None:
        rng = np.random.default_rng(42)
        adj = connect_singles(np.array([0, 0, 0]), rng)
        assert adj.nnz == 0

    def test_two_nodes_degree_one(self) -> None:
        rng = np.random.default_rng(42)
        adj = connect_singles(np.array([1, 1]), rng)
        assert adj.nnz == 2  # symmetric: (0,1) and (1,0)

    def test_odd_sum_raises(self) -> None:
        rng = np.random.default_rng(42)
        with pytest.raises(ValueError, match="even"):
            connect_singles(np.array([1, 2]), rng)


class TestPairStubsReproducibility:
    def test_same_seed_same_graph(self) -> None:
        degrees = np.array([3, 3, 3, 3, 2, 2])
        a1 = connect_singles(degrees, np.random.default_rng(42))
        a2 = connect_singles(degrees, np.random.default_rng(42))
        assert (a1 != a2).nnz == 0

    def test_different_seeds_can_differ(self) -> None:
        degrees = np.full(50, 4)
        a1 = connect_singles(degrees, np.random.default_rng(1))
        a2 = connect_singles(degrees, np.random.default_rng(2))
        assert (a1 != a2).nnz > 0


class TestPairStubsDegreePreservation:
    """Realised degrees should approximate input (lost edges from cleanup)."""

    @pytest.mark.parametrize("seed", range(10))
    def test_realised_degrees_at_most_input(self, seed: int) -> None:
        degrees = np.full(100, 6)
        adj = connect_singles(degrees, np.random.default_rng(seed))
        realised = np.asarray(adj.sum(axis=1)).flatten()
        assert np.all(realised <= degrees)

    def test_mean_degree_close_to_input(self) -> None:
        """For sparse graphs, cleanup removes few edges."""
        target = 4
        degrees = np.full(500, target)
        adj = connect_singles(degrees, np.random.default_rng(42))
        realised_mean = np.asarray(adj.sum(axis=1)).flatten().mean()
        assert abs(realised_mean - target) < 0.5
