"""Tests for ConfigModelGraph."""

import numpy as np
import pytest
from scipy.stats import poisson

from craeft.graphs.base import Subgraph, UndirectedGraph
from craeft.graphs.configuration_model import (
    ConfigModelConfig,
    ConfigModelGraph,
)
from craeft.graphs.configuration_model.connection import Connector
from craeft.graphs.configuration_model.models import DegreeMismatchError
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
        """Structurally impossible subgraph placement should exhaust retries.

        n=3 with a 3-node subgraph admits exactly one possible instance
        (there is only one way to choose all 3 of 3 labelled nodes), so
        forcing 2+ instances is a guaranteed connection-time failure on
        every retry, independent of the participation-capping fix in
        ticket 003 (which only bounds *how much* a node participates,
        not whether 3 nodes can host two distinct triangles).
        """
        from craeft.graphs.base import Subgraph  # noqa: PLC0415
        seq = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(10),  # clipped down, but still >= 2
        )
        # degree 6 comfortably covers cost=2*participation, so the
        # failure is forced at the connection step, not allocation.
        degrees = np.full(3, 6, dtype=np.int_)
        config = ConfigModelConfig(
            n=3,
            degrees=degrees,
            sequences=(seq,),
            max_retries=5,
        )
        with pytest.raises(RuntimeError, match="retries"):
            ConfigModelGraph.from_config(config, np.random.default_rng(42))

    def test_retry_exhaustion_chains_cause(self) -> None:
        """The retry-exhaustion RuntimeError must chain the real cause.

        Before ticket 003, `from_config`'s retry loop swallowed the
        underlying AllocationError/ConnectionError/ValueError, leaving
        `max_retries` exhaustion undebuggable.
        """
        from craeft.graphs.base import Subgraph  # noqa: PLC0415
        seq = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(10),
        )
        degrees = np.full(3, 6, dtype=np.int_)
        config = ConfigModelConfig(
            n=3,
            degrees=degrees,
            sequences=(seq,),
            max_retries=5,
        )
        with pytest.raises(RuntimeError) as exc_info:
            ConfigModelGraph.from_config(config, np.random.default_rng(42))

        cause = exc_info.value.__cause__
        assert cause is not None
        assert isinstance(cause, Exception)


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

    def test_allocation_error_reports_fraction_over_budget(self) -> None:
        """The error message must be self-explanatory: how many nodes,
        what fraction, the worst offender, and the mean cost vs. mean
        degree — not just the bare fact that something was exceeded.
        """
        from craeft.graphs.base import Subgraph  # noqa: PLC0415
        from craeft.graphs.configuration_model.sequence import (
            AllocationError,
            allocate_subgraphs,
        )
        seq = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(1),
        )
        # Every node costs 2*2=4 stubs but only has degree 3: all 4
        # of 4 nodes are over budget (100%).
        degrees = np.array([3, 3, 3, 3], dtype=np.int_)
        parts = np.array([2, 2, 2, 2], dtype=np.int_)
        decomp = seq._split_by_orbit(parts, np.random.default_rng(42))
        with pytest.raises(AllocationError) as exc_info:
            allocate_subgraphs(degrees, [seq], [decomp], np.random.default_rng(42))

        msg = str(exc_info.value)
        assert "4 of 4" in msg
        assert "100.0%" in msg
        assert "mean" in msg.lower()


# ---------------------------------------------------------------------------
# Degree preservation verification (ticket 002)
# ---------------------------------------------------------------------------


class TestDegreePreservationVerification:
    def test_degrees_verified_by_default(self) -> None:
        """With the matching algorithm fixed (001), generation succeeds and
        the realized degree sequence matches the target exactly."""
        rng = np.random.default_rng(7)
        degrees = np.full(200, 6, dtype=np.int_)
        config = ConfigModelConfig(n=200, degrees=degrees)
        graph = ConfigModelGraph.from_config(config, rng)
        assert np.array_equal(graph.degrees, degrees)

    def test_verify_degrees_false_allows_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verify_degrees=False returns a graph without raising, even when
        the realized degrees don't match the target."""
        monkeypatch.setattr(
            Connector, "connect_singles", lambda self, singles: None
        )
        degrees = np.full(10, 4, dtype=np.int_)
        config = ConfigModelConfig(n=10, degrees=degrees, verify_degrees=False)
        graph = ConfigModelGraph.from_config(config, np.random.default_rng(0))
        assert not np.array_equal(graph.degrees, degrees)

    def test_verify_degrees_true_raises_on_mismatch(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """verify_degrees defaults to True and raises DegreeMismatchError
        when the realized degrees don't match the target."""
        monkeypatch.setattr(
            Connector, "connect_singles", lambda self, singles: None
        )
        degrees = np.full(10, 4, dtype=np.int_)
        config = ConfigModelConfig(n=10, degrees=degrees)
        with pytest.raises(DegreeMismatchError):
            ConfigModelGraph.from_config(config, np.random.default_rng(0))

    def test_error_message_reports_node_count(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Error message names the differing node count and first bad node."""
        monkeypatch.setattr(
            Connector, "connect_singles", lambda self, singles: None
        )
        degrees = np.full(10, 4, dtype=np.int_)
        config = ConfigModelConfig(n=10, degrees=degrees)
        with pytest.raises(DegreeMismatchError) as exc_info:
            ConfigModelGraph.from_config(config, np.random.default_rng(0))
        msg = str(exc_info.value)
        assert "10 node(s)" in msg
        assert "node 0" in msg

    def test_verification_does_not_trigger_retries(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A degree mismatch surfaces on the first attempt; it is not caught
        by the retry loop and does not burn through max_retries."""
        original_init = Connector.__init__
        call_count = 0

        def counting_init(self: Connector, *args: object, **kwargs: object) -> None:
            nonlocal call_count
            call_count += 1
            original_init(self, *args, **kwargs)

        monkeypatch.setattr(Connector, "__init__", counting_init)
        monkeypatch.setattr(
            Connector, "connect_singles", lambda self, singles: None
        )

        seq = SubgraphSequence(
            subgraph=Subgraph(adjacency=TRIANGLE_ADJ),
            distribution=poisson(0),  # degenerate: always samples 0
        )
        degrees = np.full(10, 4, dtype=np.int_)
        config = ConfigModelConfig(
            n=10, degrees=degrees, sequences=(seq,), max_retries=50
        )
        with pytest.raises(DegreeMismatchError):
            ConfigModelGraph.from_config(config, np.random.default_rng(42))
        assert call_count == 1


# ---------------------------------------------------------------------------
# Ticket 003: participation must respect the degree budget by construction
# ---------------------------------------------------------------------------


def _cycle_subgraph(length: int) -> Subgraph:
    """A length-`length` cycle: every vertex has within-subgraph degree 2."""
    adjacency = np.zeros((length, length), dtype=int)
    for i in range(length):
        adjacency[i, (i + 1) % length] = 1
        adjacency[(i + 1) % length, i] = 1
    return Subgraph(adjacency=adjacency)


class TestCModelParametersBuildReliably:
    """Reproduces ticket 003's failure mode: cycle families at C-model
    parameters (n=1000, degrees=2*Pois(2), participation~Pois(2)) must
    build for every cycle length by construction, not by retry luck.

    Before the fix, this configuration built 0/300 times for every L
    at these parameters (~37% of nodes over their degree budget on a
    typical draw), because participation was sampled independently of
    each node's degree.
    """

    @pytest.mark.parametrize("length", [3, 4, 5, 6])
    def test_cmodel_parameters_build_reliably(self, length: int) -> None:
        n, lam = 1000, 2.0
        rng = np.random.default_rng(500)
        degrees = (2 * rng.poisson(lam, n)).astype(np.int_)
        if degrees.sum() % 2:
            degrees[int(np.argmax(degrees))] += 1

        seq = SubgraphSequence(
            subgraph=_cycle_subgraph(length), distribution=poisson(lam)
        )
        config = ConfigModelConfig(
            n=n,
            degrees=degrees,
            sequences=(seq,),
            max_retries=20,  # modest — success should not depend on luck
        )
        graph = ConfigModelGraph.from_config(config, rng)
        assert graph.n_nodes == n
