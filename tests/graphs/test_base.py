"""Tests for BaseGraph, UndirectedGraph, DirectedGraph, Subgraph, and DirectedSubgraph."""

from typing import Self

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from craeft.graphs.base import (
    BaseGraph,
    DirectedGraph,
    DirectedSubgraph,
    GraphConfig,
    Subgraph,
    UndirectedGraph,
)

# ---------------------------------------------------------------------------
# Concrete stubs for testing abstract base classes
# ---------------------------------------------------------------------------


class _StubUndirected(UndirectedGraph[GraphConfig]):
    @classmethod
    def from_config(cls, config: GraphConfig, rng: np.random.Generator) -> Self:
        raise NotImplementedError

    @property
    def clustering_coefficient(self) -> float:
        return 0.0


class _StubDirected(DirectedGraph[GraphConfig]):
    @classmethod
    def from_config(cls, config: GraphConfig, rng: np.random.Generator) -> Self:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Test data
# ---------------------------------------------------------------------------

TRIANGLE = csr_matrix(np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int_))

SQUARE = csr_matrix(
    np.array([[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int_)
)

DISCONNECTED = csr_matrix(
    np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=np.int_)
)

DIRECTED_CYCLE = csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int_))

DIRECTED_DISCONNECTED = csr_matrix(
    np.array([[0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int_)
)

DIRECTED_CHAIN = csr_matrix(np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=np.int_))

TRIANGLE_ADJ = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=np.int_)

SQUARE_ADJ = np.array(
    [[0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0]], dtype=np.int_
)

DIAMOND_ADJ = np.array(
    [[0, 1, 1, 1], [1, 0, 0, 1], [1, 0, 0, 1], [1, 1, 1, 0]], dtype=np.int_
)

DIRECTED_CYCLE_ADJ = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=np.int_)

DIRECTED_FAN_ADJ = np.array([[0, 1, 1], [0, 0, 1], [0, 0, 0]], dtype=np.int_)


# ---------------------------------------------------------------------------
# BaseGraph (tested via UndirectedGraph stub)
# ---------------------------------------------------------------------------


class TestBaseGraphExport:
    def test_to_csr_returns_stored_matrix(self) -> None:
        graph = _StubUndirected(TRIANGLE)
        assert (graph.to_csr() != TRIANGLE).nnz == 0

    def test_to_csr_returns_csr_type(self) -> None:
        graph = _StubUndirected(TRIANGLE)
        assert isinstance(graph.to_csr(), csr_matrix)

    def test_to_coo_preserves_data(self) -> None:
        graph = _StubUndirected(TRIANGLE)
        reconstructed = csr_matrix(graph.to_coo())
        assert (reconstructed != TRIANGLE).nnz == 0

    def test_to_coo_preserves_nnz(self) -> None:
        graph = _StubUndirected(TRIANGLE)
        assert graph.to_coo().nnz == TRIANGLE.nnz


@pytest.mark.parametrize(
    ("adjacency", "expected"),
    [
        (TRIANGLE, 3),
        (SQUARE, 4),
        (DISCONNECTED, 4),
    ],
    ids=["triangle", "square", "disconnected"],
)
class TestBaseGraphNNodes:
    def test_n_nodes(self, adjacency: csr_matrix, expected: int) -> None:
        assert _StubUndirected(adjacency).n_nodes == expected

    def test_len_equals_n_nodes(self, adjacency: csr_matrix, expected: int) -> None:
        assert len(_StubUndirected(adjacency)) == expected


class TestBaseGraphRepr:
    def test_includes_class_name(self) -> None:
        assert "_StubUndirected" in repr(_StubUndirected(TRIANGLE))

    def test_includes_counts(self) -> None:
        r = repr(_StubUndirected(TRIANGLE))
        assert "n_nodes=3" in r
        assert "n_edges=3" in r


class TestBaseGraphEq:
    def test_equal_graphs(self) -> None:
        assert _StubUndirected(TRIANGLE) == _StubUndirected(TRIANGLE)

    def test_different_graphs(self) -> None:
        assert _StubUndirected(TRIANGLE) != _StubUndirected(SQUARE)

    def test_not_implemented_for_non_graph(self) -> None:
        assert _StubUndirected(TRIANGLE).__eq__("not a graph") is NotImplemented

    def test_cross_type_equality(self) -> None:
        assert _StubUndirected(DIRECTED_CYCLE) == _StubDirected(DIRECTED_CYCLE)


# ---------------------------------------------------------------------------
# UndirectedGraph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adjacency", "expected_edges", "expected_degrees"),
    [
        (TRIANGLE, 3, [2, 2, 2]),
        (SQUARE, 4, [2, 2, 2, 2]),
        (DISCONNECTED, 2, [1, 1, 1, 1]),
    ],
    ids=["triangle", "square", "disconnected"],
)
class TestUndirectedGraphProperties:
    def test_n_edges(
        self, adjacency: csr_matrix, expected_edges: int, expected_degrees: list[int]
    ) -> None:
        assert _StubUndirected(adjacency).n_edges == expected_edges

    def test_degrees(
        self, adjacency: csr_matrix, expected_edges: int, expected_degrees: list[int]
    ) -> None:
        np.testing.assert_array_equal(
            _StubUndirected(adjacency).degrees, expected_degrees
        )


@pytest.mark.parametrize(
    ("adjacency", "expected"),
    [
        (TRIANGLE, True),
        (SQUARE, True),
        (DISCONNECTED, False),
    ],
    ids=["triangle-connected", "square-connected", "disconnected"],
)
class TestUndirectedGraphConnectivity:
    def test_is_connected(self, adjacency: csr_matrix, expected: bool) -> None:
        assert _StubUndirected(adjacency).is_connected is expected


# ---------------------------------------------------------------------------
# DirectedGraph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adjacency", "expected_edges", "expected_in", "expected_out"),
    [
        (DIRECTED_CYCLE, 3, [1, 1, 1], [1, 1, 1]),
        (DIRECTED_CHAIN, 2, [0, 1, 1], [1, 1, 0]),
        (DIRECTED_DISCONNECTED, 2, [0, 1, 0, 1], [1, 0, 1, 0]),
    ],
    ids=["cycle", "chain", "disconnected"],
)
class TestDirectedGraphProperties:
    def test_n_edges(
        self,
        adjacency: csr_matrix,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        assert _StubDirected(adjacency).n_edges == expected_edges

    def test_in_degrees(
        self,
        adjacency: csr_matrix,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        np.testing.assert_array_equal(_StubDirected(adjacency).in_degrees, expected_in)

    def test_out_degrees(
        self,
        adjacency: csr_matrix,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        np.testing.assert_array_equal(
            _StubDirected(adjacency).out_degrees, expected_out
        )


@pytest.mark.parametrize(
    ("adjacency", "expected_strong", "expected_weak"),
    [
        (DIRECTED_CYCLE, True, True),
        (DIRECTED_CHAIN, False, True),
        (DIRECTED_DISCONNECTED, False, False),
    ],
    ids=["cycle", "chain", "disconnected"],
)
class TestDirectedGraphConnectivity:
    def test_is_strongly_connected(
        self, adjacency: csr_matrix, expected_strong: bool, expected_weak: bool
    ) -> None:
        assert _StubDirected(adjacency).is_strongly_connected is expected_strong

    def test_is_weakly_connected(
        self, adjacency: csr_matrix, expected_strong: bool, expected_weak: bool
    ) -> None:
        assert _StubDirected(adjacency).is_weakly_connected is expected_weak


# ---------------------------------------------------------------------------
# Subgraph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adjacency", "expected_nodes", "expected_edges", "expected_degrees"),
    [
        (TRIANGLE_ADJ, 3, 3, [2, 2, 2]),
        (SQUARE_ADJ, 4, 4, [2, 2, 2, 2]),
        (DIAMOND_ADJ, 4, 5, [3, 2, 2, 3]),
    ],
    ids=["triangle", "square", "diamond"],
)
class TestSubgraphProperties:
    def test_num_nodes(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_degrees: list[int],
    ) -> None:
        assert Subgraph(adjacency=adjacency).num_nodes == expected_nodes

    def test_num_edges(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_degrees: list[int],
    ) -> None:
        assert Subgraph(adjacency=adjacency).num_edges == expected_edges

    def test_degrees(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_degrees: list[int],
    ) -> None:
        assert Subgraph(adjacency=adjacency).degrees == expected_degrees

    def test_len_equals_num_nodes(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_degrees: list[int],
    ) -> None:
        assert len(Subgraph(adjacency=adjacency)) == expected_nodes


class TestSubgraphEq:
    def test_equal(self) -> None:
        assert Subgraph(adjacency=TRIANGLE_ADJ) == Subgraph(
            adjacency=TRIANGLE_ADJ.copy()
        )

    def test_different(self) -> None:
        assert Subgraph(adjacency=TRIANGLE_ADJ) != Subgraph(adjacency=SQUARE_ADJ)

    def test_not_implemented_for_non_subgraph(self) -> None:
        assert (
            Subgraph(adjacency=TRIANGLE_ADJ).__eq__("not a subgraph") is NotImplemented
        )


class TestSubgraphRepr:
    def test_format(self) -> None:
        assert repr(Subgraph(adjacency=TRIANGLE_ADJ)) == "Subgraph(nodes=3, edges=3)"


# ---------------------------------------------------------------------------
# DirectedSubgraph
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("adjacency", "expected_nodes", "expected_edges", "expected_in", "expected_out"),
    [
        (DIRECTED_CYCLE_ADJ, 3, 3, [1, 1, 1], [1, 1, 1]),
        (DIRECTED_FAN_ADJ, 3, 3, [0, 1, 2], [2, 1, 0]),
    ],
    ids=["cycle", "fan"],
)
class TestDirectedSubgraphProperties:
    def test_num_nodes(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        assert DirectedSubgraph(adjacency=adjacency).num_nodes == expected_nodes

    def test_num_edges(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        assert DirectedSubgraph(adjacency=adjacency).num_edges == expected_edges

    def test_in_degrees(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        assert DirectedSubgraph(adjacency=adjacency).in_degrees == expected_in

    def test_out_degrees(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        assert DirectedSubgraph(adjacency=adjacency).out_degrees == expected_out

    def test_len_equals_num_nodes(
        self,
        adjacency: np.ndarray,
        expected_nodes: int,
        expected_edges: int,
        expected_in: list[int],
        expected_out: list[int],
    ) -> None:
        assert len(DirectedSubgraph(adjacency=adjacency)) == expected_nodes


class TestDirectedSubgraphEq:
    def test_equal(self) -> None:
        assert DirectedSubgraph(adjacency=DIRECTED_CYCLE_ADJ) == DirectedSubgraph(
            adjacency=DIRECTED_CYCLE_ADJ.copy()
        )

    def test_different(self) -> None:
        assert DirectedSubgraph(adjacency=DIRECTED_CYCLE_ADJ) != DirectedSubgraph(
            adjacency=DIRECTED_FAN_ADJ
        )

    def test_not_implemented_for_non_directed_subgraph(self) -> None:
        assert (
            DirectedSubgraph(adjacency=DIRECTED_CYCLE_ADJ).__eq__("not a subgraph")
            is NotImplemented
        )


class TestDirectedSubgraphRepr:
    def test_format(self) -> None:
        assert (
            repr(DirectedSubgraph(adjacency=DIRECTED_CYCLE_ADJ))
            == "DirectedSubgraph(nodes=3, edges=3)"
        )
