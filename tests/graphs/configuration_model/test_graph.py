"""Tests for ConfigModelGraph."""

import numpy as np
import pytest
from scipy.stats import poisson

from craeft.graphs.base import Subgraph, UndirectedGraph
from craeft.graphs.configuration_model import (
    ConfigModelConfig,
    ConfigModelGraph,
)
from craeft.graphs.configuration_model.sequence import (
    SubgraphSequence,
    sample_degree_sequence,
)

# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


class TestConfigModelConfig:
    def test_valid_config(self) -> None:
        config = ConfigModelConfig(n=4, degrees=np.array([2, 2, 2, 2]))
        assert config.n == 4

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="length"):
            ConfigModelConfig(n=3, degrees=np.array([2, 2, 2, 2]))

    def test_negative_degree_raises(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            ConfigModelConfig(n=3, degrees=np.array([2, -1, 2]))

    def test_odd_sum_raises(self) -> None:
        with pytest.raises(ValueError, match="even"):
            ConfigModelConfig(n=3, degrees=np.array([1, 1, 1]))


# ---------------------------------------------------------------------------
# Construction and type
# ---------------------------------------------------------------------------


class TestConfigModelGraphConstruction:
    def test_is_undirected_graph(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=4, degrees=np.array([2, 2, 2, 2]))
        graph = ConfigModelGraph.from_config(config, rng)
        assert isinstance(graph, UndirectedGraph)

    def test_n_nodes_matches_config(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=50, degrees=np.full(50, 4))
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.n_nodes == 50

    def test_adjacency_is_symmetric(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=30, degrees=np.full(30, 6))
        csr = ConfigModelGraph.from_config(config, rng).to_csr()
        assert (csr - csr.T).nnz == 0

    def test_no_self_loops(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=20, degrees=np.full(20, 4))
        csr = ConfigModelGraph.from_config(config, rng).to_csr()
        assert np.all(csr.diagonal() == 0)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestConfigModelGraphEdgeCases:
    def test_all_zero_degrees(self) -> None:
        rng = np.random.default_rng(42)
        graph = ConfigModelGraph.from_config(
            ConfigModelConfig(n=10, degrees=np.zeros(10, dtype=np.int_)), rng
        )
        assert graph.n_edges == 0

    def test_two_nodes_degree_one(self) -> None:
        rng = np.random.default_rng(42)
        graph = ConfigModelGraph.from_config(
            ConfigModelConfig(n=2, degrees=np.array([1, 1])), rng
        )
        assert graph.n_edges == 1


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


class TestConfigModelGraphReproducibility:
    def test_same_seed_same_graph(self) -> None:
        degrees = np.full(50, 4)
        config = ConfigModelConfig(n=50, degrees=degrees)
        g1 = ConfigModelGraph.from_config(config, np.random.default_rng(42))
        g2 = ConfigModelGraph.from_config(config, np.random.default_rng(42))
        assert g1 == g2

    def test_different_seeds_different_graphs(self) -> None:
        degrees = np.full(50, 4)
        config = ConfigModelConfig(n=50, degrees=degrees)
        g1 = ConfigModelGraph.from_config(config, np.random.default_rng(1))
        g2 = ConfigModelGraph.from_config(config, np.random.default_rng(2))
        assert g1 != g2


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


class TestConfigModelGraphProperties:
    def test_clustering_coefficient_type(self) -> None:
        rng = np.random.default_rng(42)
        config = ConfigModelConfig(n=50, degrees=np.full(50, 6))
        graph = ConfigModelGraph.from_config(config, rng)
        assert isinstance(graph.clustering_coefficient, float)

    def test_clustering_near_zero_for_sparse_graph(self) -> None:
        """Vanilla CM produces near-zero clustering."""
        rng = np.random.default_rng(42)
        degrees = sample_degree_sequence(500, poisson(5), rng)
        config = ConfigModelConfig(n=500, degrees=degrees)
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.clustering_coefficient < 0.1


# ---------------------------------------------------------------------------
# Integration: full pipeline from distribution to graph
# ---------------------------------------------------------------------------


class TestConfigModelPipeline:
    def test_sample_then_generate(self) -> None:
        rng = np.random.default_rng(42)
        degrees = sample_degree_sequence(200, poisson(4), rng)
        config = ConfigModelConfig(n=200, degrees=degrees)
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.n_nodes == 200
        assert graph.n_edges > 0

    def test_mean_degree_approximates_input(self) -> None:
        rng = np.random.default_rng(42)
        target = 6
        degrees = sample_degree_sequence(500, poisson(target), rng)
        config = ConfigModelConfig(n=500, degrees=degrees)
        graph = ConfigModelGraph.from_config(config, rng)
        assert abs(graph.degrees.mean() - target) < 1.0


# ---------------------------------------------------------------------------
# Subgraph sequence pipeline
# ---------------------------------------------------------------------------


TRIANGLE_ADJ = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]])


def _tri_seq(lam: float = 0.3) -> SubgraphSequence:
    from craeft.graphs.base import Subgraph  # noqa: PLC0415
    return SubgraphSequence(
        subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
        distribution=poisson(lam),
    )


class TestConfigModelWithSubgraphs:
    """End-to-end generation with subgraph sequences."""

    def test_triangle_sequence_generates_valid_graph(self) -> None:
        rng = np.random.default_rng(42)
        degrees = np.full(30, 4, dtype=np.int_)
        config = ConfigModelConfig(
            n=30,
            degrees=degrees,
            sequences=(_tri_seq(0.25),),
            max_retries=200,
        )
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.n_nodes == 30
        assert graph.n_edges > 0
        assert isinstance(graph, UndirectedGraph)
        csr = graph.to_csr()
        assert (csr - csr.T).nnz == 0
        assert np.all(csr.diagonal() == 0)

    def test_triangle_produces_higher_clustering(self) -> None:
        """With enough triangles, clustering should exceed vanilla CM."""
        rng = np.random.default_rng(99)
        n = 40
        tri_seq_strong = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(0.5),
        )
        degrees = np.full(n, 5, dtype=np.int_)

        clustered = ConfigModelGraph.from_config(
            ConfigModelConfig(
                n=n,
                degrees=degrees,
                sequences=(tri_seq_strong,),
                max_retries=200,
            ),
            rng,
        )
        # With triangle subgraphs, clustering should be measurably higher
        # than the near-zero clustering of vanilla CM
        assert clustered.clustering_coefficient > 0.05

    def test_reproducibility_with_sequences(self) -> None:
        degrees = np.full(20, 4, dtype=np.int_)
        config = ConfigModelConfig(
            n=20,
            degrees=degrees,
            sequences=(_tri_seq(),),
            max_retries=200,
        )
        g1 = ConfigModelGraph.from_config(config, np.random.default_rng(42))
        g2 = ConfigModelGraph.from_config(config, np.random.default_rng(42))
        assert g1 == g2

    def test_retry_exhaustion_raises(self) -> None:
        """Impossibly high participation should exhaust retries."""
        from craeft.graphs.base import Subgraph  # noqa: PLC0415
        seq = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(10),  # way too many
        )
        degrees = np.full(6, 3, dtype=np.int_)  # degree 3 can't support 10*2 stubs
        config = ConfigModelConfig(
            n=6,
            degrees=degrees,
            sequences=(seq,),
            max_retries=5,
        )
        with pytest.raises(RuntimeError, match="retries"):
            ConfigModelGraph.from_config(config, np.random.default_rng(42))


class TestAllocationErrorDetection:
    """Allocation correctly detects degree budget violations."""

    def test_budget_exceeded_raises(self) -> None:
        from craeft.graphs.base import Subgraph  # noqa: PLC0415
        from craeft.graphs.configuration_model.sequence import (
            AllocationError,
            allocate_subgraphs,
        )
        seq = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(1),
        )
        # Decomposition requires 2*2=4 stubs per node, but degree is only 3
        degrees = np.array([3, 3, 3, 3], dtype=np.int_)
        parts = np.array([2, 2, 2, 2], dtype=np.int_)  # 4 nodes * 2 = 8 total
        decomp = seq._split_by_orbit(parts, np.random.default_rng(42))
        with pytest.raises(AllocationError, match="exceeded"):
            allocate_subgraphs(degrees, [seq], [decomp], np.random.default_rng(42))

    def test_prescribed_split_over_budget_raises(self) -> None:
        """A prescribed orbit split that exceeds a node's degree budget
        must surface via AllocationError, not be silently repaired
        (ticket 007)."""
        from craeft.graphs.base import Subgraph  # noqa: PLC0415
        from craeft.graphs.configuration_model.sequence import (
            AllocationError,
            allocate_subgraphs,
        )
        # Triangle is vertex-transitive: single orbit 0, sigma_0=3.
        # Total=9 -> M=3 whole instances, consistent, passes validation.
        counts = {0: np.array([3, 3, 3, 0], dtype=np.int_)}
        seq = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(1),
            orbit_counts=counts,
        )
        # Cost per node = count * orbit_degree(2) = [6, 6, 6, 0] > degree 3.
        degrees = np.array([3, 3, 3, 3], dtype=np.int_)
        dummy_parts = np.array([1, 1, 1, 1], dtype=np.int_)  # ignored
        decomp = seq._split_by_orbit(dummy_parts, np.random.default_rng(0))
        with pytest.raises(AllocationError, match="exceeded"):
            allocate_subgraphs(degrees, [seq], [decomp], np.random.default_rng(0))
