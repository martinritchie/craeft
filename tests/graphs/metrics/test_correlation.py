"""Tests for degree-degree correlation and degree-conditioned metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from craeft.graphs.metrics import (
    average_neighbour_degree,
    clustering_by_degree,
    degree_assortativity,
)


def _from_edges(n: int, edges: list[tuple[int, int]]) -> csr_matrix:
    rows = [e[0] for e in edges] + [e[1] for e in edges]
    cols = [e[1] for e in edges] + [e[0] for e in edges]
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n))


def _ring(n: int) -> csr_matrix:
    edges = [(i, (i + 1) % n) for i in range(n)]
    return _from_edges(n, edges)


def _star(n: int) -> csr_matrix:
    edges = [(0, i) for i in range(1, n)]
    return _from_edges(n, edges)


def _random_edges(n: int, m: int, seed: int) -> list[tuple[int, int]]:
    """m distinct undirected edges over n nodes, no self-loops, seeded."""
    rng = np.random.default_rng(seed)
    edges: set[tuple[int, int]] = set()
    while len(edges) < m:
        i, j = rng.integers(0, n, size=2)
        if i == j:
            continue
        edges.add((min(int(i), int(j)), max(int(i), int(j))))
    return sorted(edges)


def _random_regular_like(n: int, seed: int) -> csr_matrix:
    """A moderately heterogeneous random graph for range/igraph cross-checks."""
    return _from_edges(n, _random_edges(n, m=int(n * 2.5), seed=seed))


# ---------------------------------------------------------------------------
# degree_assortativity
# ---------------------------------------------------------------------------


class TestDegreeAssortativity:
    def test_regular_ring_is_nan(self) -> None:
        adj = _ring(10)
        assert np.isnan(degree_assortativity(adj))

    def test_star_is_negative(self) -> None:
        adj = _star(10)
        r = degree_assortativity(adj)
        assert r < -0.5

    def test_empty_graph_is_nan(self) -> None:
        adj = csr_matrix((5, 5), dtype=np.float64)
        assert np.isnan(degree_assortativity(adj))

    @pytest.mark.parametrize("seed", range(5))
    def test_matches_igraph(self, seed: int) -> None:
        import igraph as ig  # noqa: PLC0415

        edges = _random_edges(60, m=200, seed=seed)
        adj = _from_edges(60, edges)
        g = ig.Graph(n=60, edges=edges, directed=False)
        mine = degree_assortativity(adj)
        theirs = g.assortativity_degree()
        assert mine == pytest.approx(theirs, abs=1e-9)

    @pytest.mark.parametrize("seed", range(5))
    def test_in_range(self, seed: int) -> None:
        adj = _random_regular_like(60, seed)
        r = degree_assortativity(adj)
        assert -1.0 <= r <= 1.0

    def test_accepts_csr_and_dense_compatible_input(self) -> None:
        adj = csr_matrix(_star(6))
        assert not np.isnan(degree_assortativity(adj))


# ---------------------------------------------------------------------------
# average_neighbour_degree
# ---------------------------------------------------------------------------


class TestAverageNeighbourDegree:
    def test_shape_matches_n_nodes(self) -> None:
        adj = _random_regular_like(30, 1)
        result = average_neighbour_degree(adj)
        assert result.shape == (30,)

    def test_ring_all_equal_to_two(self) -> None:
        adj = _ring(8)
        result = average_neighbour_degree(adj)
        np.testing.assert_array_almost_equal(result, np.full(8, 2.0))

    def test_star_leaves_have_hub_degree(self) -> None:
        n = 6
        adj = _star(n)
        result = average_neighbour_degree(adj)
        # Leaves' only neighbour is the hub (degree n - 1)
        np.testing.assert_array_almost_equal(result[1:], np.full(n - 1, n - 1))

    def test_isolated_node_is_zero(self) -> None:
        adj = _from_edges(4, [(0, 1)])
        result = average_neighbour_degree(adj)
        assert result[2] == 0.0
        assert result[3] == 0.0


# ---------------------------------------------------------------------------
# clustering_by_degree
# ---------------------------------------------------------------------------


class TestClusteringByDegree:
    def test_keys_are_degree_values_present(self) -> None:
        adj = _random_regular_like(40, 2)
        degrees = np.asarray(adj.sum(axis=1)).ravel().astype(int)
        result = clustering_by_degree(adj)
        assert set(result.keys()) == set(degrees.tolist())

    def test_triangle_degree_two_has_clustering_one(self) -> None:
        adj = _from_edges(3, [(0, 1), (1, 2), (2, 0)])
        result = clustering_by_degree(adj)
        assert result == {2: pytest.approx(1.0)}

    def test_star_two_classes(self) -> None:
        n = 5
        adj = _star(n)
        result = clustering_by_degree(adj)
        assert set(result.keys()) == {1, n - 1}
        assert result[1] == pytest.approx(0.0)
        assert result[n - 1] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Hub-concentrated clustered subgraphs raise assortativity (2017 Fig. 7 result)
# ---------------------------------------------------------------------------


class TestHubConcentratedSubgraphsRaiseAssortativity:
    def _hub_leaf_base(
        self, n_hubs: int, n_leaves_per_hub: int
    ) -> tuple[csr_matrix, list[int], list[int]]:
        """Disassortative hub-and-spoke base: hubs each own a private leaf set."""
        n = n_hubs + n_hubs * n_leaves_per_hub
        hubs = list(range(n_hubs))
        edges = []
        leaf_id = n_hubs
        leaves: list[int] = []
        for hub in hubs:
            for _ in range(n_leaves_per_hub):
                edges.append((hub, leaf_id))
                leaves.append(leaf_id)
                leaf_id += 1
        return _from_edges(n, edges), hubs, leaves

    def test_hub_triangles_more_assortative_than_leaf_triangles(self) -> None:
        base_adj, hubs, leaves = self._hub_leaf_base(n_hubs=4, n_leaves_per_hub=10)
        n = base_adj.shape[0]

        # Variant A: extra clustered edges placed among low-degree leaves
        # (unconstrained placement).
        leaf_extra_edges = [
            (leaves[0], leaves[1]),
            (leaves[1], leaves[2]),
            (leaves[2], leaves[0]),
            (leaves[3], leaves[4]),
            (leaves[4], leaves[5]),
            (leaves[5], leaves[3]),
        ]
        leaf_adj = _add_edges(base_adj, n, leaf_extra_edges)

        # Variant B: extra clustered edges placed among high-degree hubs
        # (hub-concentrated placement) — same edge count.
        hub_extra_edges = [
            (hubs[0], hubs[1]),
            (hubs[1], hubs[2]),
            (hubs[2], hubs[0]),
            (hubs[0], hubs[3]),
            (hubs[1], hubs[3]),
            (hubs[2], hubs[3]),
        ]
        hub_adj = _add_edges(base_adj, n, hub_extra_edges)

        r_leaf = degree_assortativity(leaf_adj)
        r_hub = degree_assortativity(hub_adj)

        assert r_hub > r_leaf


def _add_edges(adj: csr_matrix, n: int, extra: list[tuple[int, int]]) -> csr_matrix:
    coo = adj.tocoo()
    rows = coo.row.tolist() + [e[0] for e in extra] + [e[1] for e in extra]
    cols = coo.col.tolist() + [e[1] for e in extra] + [e[0] for e in extra]
    data = np.ones(len(rows), dtype=np.float64)
    return csr_matrix((data, (rows, cols)), shape=(n, n))
