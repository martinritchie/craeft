"""Tests for connected component analysis."""

import numpy as np
import pytest
from scipy.sparse import csr_array, csr_matrix

from craeft.networks.metrics import (
    DisconnectedGraphError,
    is_connected,
)


class TestIsConnected:
    """Tests for is_connected."""

    def test_triangle_is_connected(self, triangle_graph: csr_array) -> None:
        assert is_connected(triangle_graph) is True

    def test_complete_k4_is_connected(self, complete_graph_k4: csr_array) -> None:
        assert is_connected(complete_graph_k4) is True

    def test_chain_is_connected(self, chain_graph: csr_array) -> None:
        assert is_connected(chain_graph) is True

    def test_star_is_connected(self, star_graph: csr_array) -> None:
        assert is_connected(star_graph) is True

    def test_empty_graph_is_not_connected(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.float64)
        assert is_connected(adj) is False

    def test_two_components(self) -> None:
        # Two disconnected edges: 0-1 and 2-3
        dense = np.array(
            [
                [0, 1, 0, 0],
                [1, 0, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=np.float64,
        )
        assert is_connected(csr_array(dense)) is False

    def test_single_node(self) -> None:
        adj = csr_matrix((1, 1), dtype=np.float64)
        assert is_connected(adj) is True

    def test_single_edge(self) -> None:
        adj = csr_matrix([[0, 1], [1, 0]], dtype=np.float64)
        assert is_connected(adj) is True


class TestDisconnectedGraphError:
    """Tests for the custom exception."""

    def test_is_runtime_error(self) -> None:
        assert issubclass(DisconnectedGraphError, RuntimeError)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(DisconnectedGraphError, match="not connected"):
            raise DisconnectedGraphError("Graph is not connected")
