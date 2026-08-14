"""Tests for designed clustering: unique triangles and closed-form design metrics."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import poisson, randint

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model import ConfigModelConfig
from craeft.graphs.configuration_model.sequence import SubgraphSequence
from craeft.graphs.metrics import (
    designed_clustering,
    designed_triangles,
    triangles_per_orbit,
    unique_triangles,
)

# ---------------------------------------------------------------------------
# Known subgraph patterns
# ---------------------------------------------------------------------------


def _edge() -> Subgraph:
    return Subgraph(adjacency=np.array([[0, 1], [1, 0]]))


def _triangle() -> Subgraph:
    return Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))


def _cycle(n: int) -> Subgraph:
    adj = np.zeros((n, n), dtype=int)
    for i in range(n):
        adj[i, (i + 1) % n] = 1
        adj[(i + 1) % n, i] = 1
    return Subgraph(adjacency=adj)


def _diamond() -> Subgraph:
    """K4 minus one edge. Tips (0, 2) degree 2, hubs (1, 3) degree 3."""
    return Subgraph(
        adjacency=np.array(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 1],
                [0, 1, 0, 1],
                [1, 1, 1, 0],
            ]
        )
    )


def _complete(n: int) -> Subgraph:
    return Subgraph(adjacency=np.ones((n, n), dtype=int) - np.eye(n, dtype=int))


# ---------------------------------------------------------------------------
# unique_triangles
# ---------------------------------------------------------------------------


class TestUniqueTriangles:
    @pytest.mark.parametrize(
        ("subgraph_fn", "expected"),
        [
            (_edge, 0),
            (_triangle, 1),
            (lambda: _cycle(4), 0),
            (lambda: _cycle(5), 0),
            (lambda: _cycle(6), 0),
            (_diamond, 2),
            (lambda: _complete(4), 4),
            (lambda: _complete(5), 10),
            (lambda: _complete(6), 20),
        ],
        ids=[
            "edge",
            "triangle",
            "4-cycle",
            "5-cycle",
            "6-cycle",
            "diamond",
            "K4",
            "K5",
            "K6",
        ],
    )
    def test_known_subgraphs(self, subgraph_fn, expected: int) -> None:
        assert unique_triangles(subgraph_fn()) == expected


# ---------------------------------------------------------------------------
# triangles_per_orbit
# ---------------------------------------------------------------------------


class TestTrianglesPerOrbit:
    def test_diamond(self) -> None:
        seq = SubgraphSequence(subgraph=_diamond(), distribution=randint(1, 2))
        result = triangles_per_orbit(seq)
        assert result == {0: 1, 1: 2}

    def test_triangle_single_orbit(self) -> None:
        seq = SubgraphSequence(subgraph=_triangle(), distribution=randint(1, 2))
        result = triangles_per_orbit(seq)
        assert result == {0: 1}

    def test_4cycle_all_zero(self) -> None:
        seq = SubgraphSequence(subgraph=_cycle(4), distribution=randint(1, 2))
        result = triangles_per_orbit(seq)
        assert result == {0: 0}


# ---------------------------------------------------------------------------
# designed_triangles / designed_clustering
# ---------------------------------------------------------------------------


def _config_with_sequences(
    n: int,
    degree: int,
    sequences: tuple[SubgraphSequence, ...],
) -> ConfigModelConfig:
    degrees = np.full(n, degree, dtype=np.int_)
    return ConfigModelConfig(n=n, degrees=degrees, sequences=sequences)


class TestDesignedTriangles:
    def test_matches_hand_calculation(self) -> None:
        n = 300
        rate = 3  # constant participation per node (prescribed, deterministic)
        dist = randint(rate, rate + 1)
        seq = SubgraphSequence(subgraph=_triangle(), distribution=dist)
        config = _config_with_sequences(n, degree=10, sequences=(seq,))

        expected_instances = n * rate / 3
        expected_triangles = expected_instances * 1  # unique_triangles(triangle) == 1
        assert designed_triangles(config) == pytest.approx(expected_triangles)

    def test_zero_for_empty_cycle_sequences(self) -> None:
        for k in (4, 5, 6):
            seq = SubgraphSequence(subgraph=_cycle(k), distribution=randint(1, 2))
            config = _config_with_sequences(n=300, degree=10, sequences=(seq,))
            assert designed_triangles(config) == pytest.approx(0.0)

    def test_zero_with_no_sequences(self) -> None:
        config = _config_with_sequences(n=50, degree=4, sequences=())
        assert designed_triangles(config) == 0.0


class TestDesignedClustering:
    def test_matches_hand_calculation(self) -> None:
        n = 300
        rate = 3
        dist = randint(rate, rate + 1)
        seq = SubgraphSequence(subgraph=_triangle(), distribution=dist)
        config = _config_with_sequences(n, degree=10, sequences=(seq,))

        m = n * rate / 3
        k = np.full(n, 10)
        triples = int((k * (k - 1) // 2).sum())
        expected = 3 * m / triples
        assert designed_clustering(config) == pytest.approx(expected)

    def test_zero_for_empty_cycles(self) -> None:
        for k in (4, 5, 6):
            seq = SubgraphSequence(subgraph=_cycle(k), distribution=randint(1, 2))
            config = _config_with_sequences(n=300, degree=10, sequences=(seq,))
            assert designed_clustering(config) == 0.0

    def test_no_sequences_is_zero(self) -> None:
        config = _config_with_sequences(n=50, degree=4, sequences=())
        assert designed_clustering(config) == 0.0

    def test_zero_degree_sequence_is_zero(self) -> None:
        config = ConfigModelConfig(n=10, degrees=np.zeros(10, dtype=np.int_))
        assert designed_clustering(config) == 0.0


# ---------------------------------------------------------------------------
# designed vs realized (build and compare)
# ---------------------------------------------------------------------------


class TestDesignedWithinToleranceOfRealized:
    def test_realized_close_to_designed_plus_floor(self) -> None:
        from craeft.graphs.configuration_model import ConfigModelGraph
        from craeft.graphs.metrics import count_triangles

        n = 800
        degree = 10
        tri_seq = SubgraphSequence(subgraph=_triangle(), distribution=poisson(0.3))
        degrees = np.full(n, degree, dtype=np.int_)

        clustered_config = ConfigModelConfig(
            n=n, degrees=degrees, sequences=(tri_seq,), max_retries=200
        )
        baseline_config = ConfigModelConfig(n=n, degrees=degrees)

        clustered_graph = ConfigModelGraph.from_config(
            clustered_config, np.random.default_rng(7)
        )
        baseline_graph = ConfigModelGraph.from_config(
            baseline_config, np.random.default_rng(7)
        )

        designed = designed_triangles(clustered_config)
        floor = count_triangles(baseline_graph.to_csr())
        realized = count_triangles(clustered_graph.to_csr())

        predicted = designed + floor
        # Single-realization stochastic comparison: allow generous tolerance
        # (paper's own control audit measured 0.2-2.2% on averaged runs).
        assert realized == pytest.approx(predicted, rel=0.2)


# ---------------------------------------------------------------------------
# ConfigModelGraph.designed_clustering property
# ---------------------------------------------------------------------------


class TestGraphDesignedClusteringProperty:
    def test_property_matches_function(self) -> None:
        from craeft.graphs.configuration_model import ConfigModelGraph

        n = 100
        seq = SubgraphSequence(subgraph=_triangle(), distribution=poisson(0.2))
        config = _config_with_sequences(n, degree=8, sequences=(seq,))
        graph = ConfigModelGraph.from_config(config, np.random.default_rng(3))
        assert graph.designed_clustering == pytest.approx(designed_clustering(config))

    def test_raises_without_config(self) -> None:
        from scipy.sparse import csr_matrix

        from craeft.graphs.configuration_model import ConfigModelGraph

        adj = csr_matrix(np.array([[0, 1], [1, 0]]))
        graph = ConfigModelGraph(adj)
        with pytest.raises(ValueError, match="from_config"):
            _ = graph.designed_clustering
