"""Tests for assembly module."""

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from craeft.networks.generation.configuration_model.assembly import (
    AssemblyResult,
    assemble,
)
from craeft.networks.generation.configuration_model.pairing import PairingResult


@pytest.fixture
def empty_pairing() -> PairingResult:
    return PairingResult(
        rows=np.array([], dtype=np.int_),
        cols=np.array([], dtype=np.int_),
        unpaired=0,
    )


@pytest.fixture
def empty_motif_edges() -> tuple[np.ndarray, np.ndarray]:
    return np.array([], dtype=np.int_), np.array([], dtype=np.int_)


class TestAssemblyResult:
    def test_num_edges_from_adjacency(self) -> None:
        """num_edges returns half the nnz (undirected count)."""
        adj = csr_matrix(([1, 1, 1, 1], ([0, 1, 0, 2], [1, 0, 2, 0])), shape=(3, 3))
        result = AssemblyResult(adjacency=adj, num_motif_edges=1, num_single_edges=1)
        assert result.num_edges == 2

    def test_frozen_dataclass(self) -> None:
        """AssemblyResult is immutable."""
        adj = csr_matrix((3, 3))
        result = AssemblyResult(adjacency=adj, num_motif_edges=0, num_single_edges=0)
        with pytest.raises(AttributeError):
            result.num_motif_edges = 5  # type: ignore[misc]


class TestAssemble:
    def test_empty_edges(
        self,
        empty_motif_edges: tuple[np.ndarray, np.ndarray],
        empty_pairing: PairingResult,
    ) -> None:
        """No edges produces empty adjacency."""
        result = assemble(5, empty_motif_edges, empty_pairing)
        assert result.adjacency.nnz == 0
        assert result.num_edges == 0
        assert result.num_motif_edges == 0
        assert result.num_single_edges == 0

    def test_motif_edges_only(self, empty_pairing: PairingResult) -> None:
        """Motif edges without single edges."""
        motif_edges = (np.array([0, 0, 1]), np.array([1, 2, 2]))  # Triangle
        result = assemble(3, motif_edges, empty_pairing)

        assert result.num_edges == 3
        assert result.num_motif_edges == 3
        assert result.num_single_edges == 0
        assert result.adjacency.shape == (3, 3)

    def test_single_edges_only(
        self,
        empty_motif_edges: tuple[np.ndarray, np.ndarray],
    ) -> None:
        """Single edges without motif edges."""
        pairing = PairingResult(
            rows=np.array([0, 2]),
            cols=np.array([1, 3]),
            unpaired=0,
        )
        result = assemble(4, empty_motif_edges, pairing)

        assert result.num_edges == 2
        assert result.num_motif_edges == 0
        assert result.num_single_edges == 2

    def test_combined_edges(self) -> None:
        """Motif and single edges combined correctly."""
        motif_edges = (np.array([0, 0, 1]), np.array([1, 2, 2]))  # Triangle 0-1-2
        pairing = PairingResult(
            rows=np.array([3, 4]),
            cols=np.array([4, 5]),
            unpaired=0,
        )
        result = assemble(6, motif_edges, pairing)

        assert result.num_edges == 5
        assert result.num_motif_edges == 3
        assert result.num_single_edges == 2

    def test_adjacency_is_symmetric(self) -> None:
        """Adjacency matrix is symmetric."""
        motif_edges = (np.array([0, 1]), np.array([1, 2]))
        pairing = PairingResult(
            rows=np.array([2]),
            cols=np.array([3]),
            unpaired=0,
        )
        result = assemble(4, motif_edges, pairing)

        adj = result.adjacency.toarray()
        np.testing.assert_array_equal(adj, adj.T)

    def test_no_self_loops(self) -> None:
        """Self-loops in input are removed."""
        motif_edges = (np.array([0, 1, 2]), np.array([0, 2, 3]))  # (0,0) is self-loop
        pairing = PairingResult(
            rows=np.array([4]),
            cols=np.array([4]),  # Also self-loop
            unpaired=0,
        )
        result = assemble(5, motif_edges, pairing)

        adj = result.adjacency.toarray()
        assert all(adj[i, i] == 0 for i in range(5))

    def test_multi_edges_collapsed(self) -> None:
        """Duplicate edges are collapsed to single edges."""
        # Same edge twice in motif
        motif_edges = (np.array([0, 0]), np.array([1, 1]))
        pairing = PairingResult(
            rows=np.array([0]),  # Another (0,1)
            cols=np.array([1]),
            unpaired=0,
        )
        result = assemble(2, motif_edges, pairing)

        adj = result.adjacency.toarray()
        assert adj[0, 1] == 1
        assert adj[1, 0] == 1
        assert result.num_edges == 1

    def test_adjacency_dtype_int8(self) -> None:
        """Adjacency uses int8 for memory efficiency."""
        motif_edges = (np.array([0]), np.array([1]))
        pairing = PairingResult(
            rows=np.array([], dtype=np.int_),
            cols=np.array([], dtype=np.int_),
            unpaired=0,
        )
        result = assemble(2, motif_edges, pairing)
        assert result.adjacency.dtype == np.int8

    def test_large_node_ids(self) -> None:
        """Handles large node IDs correctly."""
        motif_edges = (np.array([100, 100, 200]), np.array([200, 300, 300]))
        pairing = PairingResult(
            rows=np.array([400]),
            cols=np.array([500]),
            unpaired=0,
        )
        result = assemble(1000, motif_edges, pairing)

        assert result.adjacency.shape == (1000, 1000)
        assert result.num_edges == 4
        # Verify specific edges exist
        adj = result.adjacency
        assert adj[100, 200] == 1
        assert adj[400, 500] == 1
