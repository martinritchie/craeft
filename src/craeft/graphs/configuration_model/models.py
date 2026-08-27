"""Graph config and class for the configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.graphs.base import GraphConfig, UndirectedGraph
from craeft.graphs.configuration_model.connection import (
    ConnectionError,
    Connector,
)
from craeft.graphs.configuration_model.sequence import (
    AllocationError,
    SubgraphSequence,
    allocate_subgraphs,
)
from craeft.graphs.metrics.clustering import global_clustering_coefficient
from craeft.graphs.metrics.subgraph import designed_clustering as _designed_clustering


@dataclass(frozen=True)
class ConfigModelConfig(GraphConfig):
    """Configuration for a configuration model graph.

    Attributes:
        n: Number of nodes.
        degrees: Per-node degree sequence. Length must equal n,
            values non-negative, sum must be even.
        sequences: Subgraph sequences to embed. When empty, produces
            a standard configuration model graph (edge-only).
        max_retries: Maximum reset-and-retry attempts when subgraph
            allocation or connection fails.
        verify_degrees: Raise if the realized degree sequence differs
            from ``degrees``.

            Set False only for exploratory work where approximate
            degrees are acceptable.
    """

    degrees: NDArray[np.int_]
    sequences: tuple[SubgraphSequence, ...] = ()
    max_retries: int = 100
    verify_degrees: bool = True

    def __post_init__(self) -> None:
        if len(self.degrees) != self.n:
            msg = f"Degree sequence length {len(self.degrees)} != n ({self.n})"
            raise ValueError(msg)
        if np.any(self.degrees < 0):
            msg = "Degrees must be non-negative"
            raise ValueError(msg)
        if int(self.degrees.sum()) % 2 != 0:
            msg = f"Degree sum must be even, got {int(self.degrees.sum())}"
            raise ValueError(msg)


class DegreeMismatchError(Exception):
    """Raised when the realized degree sequence differs from the prescribed one."""


# How many offending nodes to name before truncating the message.
_MAX_NODES_IN_ERROR = 5


def _preflight_prescribed(config: ConfigModelConfig) -> None:
    """Validate prescribed orbit splits once, before the retry loop.

    ``SubgraphSequence.__post_init__`` can only check what a sequence
    knows about itself — orbit keys, equal array lengths, per-orbit
    totals of the form ``M * sigma_o``. Two things it cannot see are
    ``n`` and the degree sequence, and both are only known here.

    Placement matters as much as the checks. A prescription is
    *deterministic*: if it is the wrong length or over budget, it is
    wrong on every attempt. Letting such a failure fall into the retry
    loop — which exists for *stochastic* dead ends — spends all
    ``max_retries`` attempts reaching the same conclusion and then
    reports a generic ``RuntimeError``, burying the real diagnosis in
    ``__cause__``. Running once, up front, keeps the error legible.

    Args:
        config: Configuration model configuration.

    Raises:
        ValueError: If a prescribed count array's length differs
            from ``config.n``.
        AllocationError: If the prescribed sequences alone exceed any
            node's degree budget.

    Note:
        The budget check covers the *prescribed* sequences only, so
        when sampled sequences are also present it is necessary but
        not sufficient — passing it does not guarantee the combined
        cost fits. ``allocate_subgraphs`` remains the authoritative
        check inside the loop. Failing it, though, is conclusive:
        the prescription cannot fit no matter what is sampled.
    """
    prescribed = [
        (idx, seq)
        for idx, seq in enumerate(config.sequences)
        if seq.orbit_counts is not None
    ]
    if not prescribed:
        return

    for idx, seq in prescribed:
        assert seq.orbit_counts is not None  # guarded by the filter above
        for orbit, counts in seq.orbit_counts.items():
            if len(counts) != config.n:
                msg = (
                    f"Prescribed orbit_counts for sequence {idx}, orbit "
                    f"{orbit} has length {len(counts)}, but n is {config.n}. "
                    "A prescription must give a count for every node."
                )
                raise ValueError(msg)

    cost = np.zeros(config.n, dtype=np.int_)
    for _, seq in prescribed:
        assert seq.orbit_counts is not None
        for orbit, counts in seq.orbit_counts.items():
            cost += np.asarray(counts, dtype=np.int_) * seq.orbit_degrees[orbit]

    over = np.flatnonzero(cost > config.degrees)
    if over.size == 0:
        return

    named = ", ".join(
        f"node {int(i)} needs {int(cost[i])} but has {int(config.degrees[i])}"
        for i in over[:_MAX_NODES_IN_ERROR]
    )
    if over.size > _MAX_NODES_IN_ERROR:
        named += f", and {int(over.size) - _MAX_NODES_IN_ERROR} more"
    msg = (
        f"Prescribed orbit_counts exceed the degree budget for "
        f"{int(over.size)} of {config.n} node(s) "
        f"({over.size / config.n:.1%}): {named}. "
        "A prescribed split is deterministic, so retrying cannot help "
        "and it is not silently repaired — adjust the prescription or "
        "the degree sequence."
    )
    raise AllocationError(msg)


def _verify_degrees(graph: ConfigModelGraph, config: ConfigModelConfig) -> None:
    """Raise DegreeMismatchError if realized degrees differ from config.degrees.

    No-op when config.verify_degrees is False.
    """
    if not config.verify_degrees:
        return
    realized = graph.degrees
    if np.array_equal(realized, config.degrees):
        return
    diff = realized - config.degrees
    bad = int(np.count_nonzero(diff))
    msg = (
        f"Degree sequence not preserved: {bad} node(s) differ "
        f"(total |deficit| {int(np.abs(diff).sum())}, "
        f"max {int(np.abs(diff).max())}). "
        f"First at node {int(np.flatnonzero(diff)[0])}."
    )
    raise DegreeMismatchError(msg)


class ConfigModelGraph(UndirectedGraph[ConfigModelConfig]):
    """Random graph with a prescribed degree sequence and optional
    subgraph structure.

    When no subgraph sequences are specified, generates a standard
    configuration model graph via stub pairing. When sequences are
    provided, splits them by orbit, greedily allocates participations
    to nodes, connects subgraph instances (checking for duplicates
    and existing edges), then pairs remaining single stubs.

    Retries from scratch on failure (dead-end configurations).
    """

    def __init__(
        self,
        adjacency: csr_matrix,
        config: ConfigModelConfig | None = None,
    ) -> None:
        super().__init__(adjacency)
        self._config = config

    @classmethod
    def from_config(
        cls,
        config: ConfigModelConfig,
        rng: np.random.Generator,
    ) -> Self:
        """Generate a configuration model graph.

        Without subgraph sequences:
            1. Pair stubs from the degree sequence
            2. Remove self-loops and multi-edges

        With subgraph sequences:
            1. Sample participation sequences
            2. Split each by orbit
            3. Greedily allocate to nodes
            4. Connect subgraph instances (with multi-edge check)
            5. Pair remaining single stubs
            6. Assemble into adjacency matrix

        On failure (AllocationError or ConnectionError), retries
        up to config.max_retries times with fresh random state.

        Args:
            config: Configuration model configuration.
            rng: Random number generator.

        Returns:
            A ConfigModelGraph instance.

        Raises:
            RuntimeError: If all retries exhausted.
            ValueError: If a prescribed ``orbit_counts`` array's length
                differs from ``config.n`` (raised by the pre-flight
                check, before the retry loop).
            AllocationError: If prescribed sequences alone exceed the
                degree budget. Also from the pre-flight check: a
                prescription is deterministic, so this is settled once
                rather than rediscovered on every retry.
            DegreeMismatchError: If config.verify_degrees is True and the
                realized degree sequence differs from config.degrees. Not
                caught by the retry loop above: a mismatch is a correctness
                bug, not a dead-end configuration worth retrying.
        """
        if not config.sequences:
            connector = Connector(config.n, rng)
            connector.connect_singles(config.degrees)
            graph = cls(connector.to_csr(), config=config)
            _verify_degrees(graph, config)
            return graph

        # Anything deterministic is settled before the loop, so a bad
        # prescription raises its own diagnosis rather than max_retries
        # copies of it wrapped in a RuntimeError.
        _preflight_prescribed(config)

        # Subgraph sequence pipeline
        last_exc: Exception | None = None
        for _ in range(config.max_retries):
            try:
                connector = Connector(config.n, rng)
                decompositions: list[dict[int, NDArray[np.int_]]] = []

                # 1. Sample participation sequences and split by orbit
                #
                # Ticket 003: sampling participation with no reference to
                # each node's degree budget fails almost surely once the
                # participation distribution has spread comparable to the
                # degree distribution — the allocation check below would
                # reject nearly every draw. We derive a conservative
                # per-node cap from the *worst-case* stub cost: a node's
                # eventual orbit split isn't known until after sampling
                # (``_split_by_orbit`` runs next), so we can't know which
                # orbit a given participation will land in. Using the
                # subgraph's most expensive orbit (max(orbit_degrees))
                # as the per-participation cost is always safe — no
                # possible orbit split can then exceed the node's degree.
                # This is conservative rather than exact: when a sequence
                # is not vertex-transitive, cheaper orbits will
                # under-use the true budget, and when multiple sequences
                # share one degree budget, each sequence is capped
                # independently against the *full* degree rather than a
                # fair share of it. Both are acceptable because
                # ``allocate_subgraphs`` still performs the authoritative
                # cross-sequence check and raises ``AllocationError``
                # (caught below, triggering a retry) if the combined
                # cost from multiple sequences is still too high — this
                # cap only needs to fix the dominant single-sequence
                # failure mode, not guarantee success in every case.
                for seq in config.sequences:
                    if seq.orbit_counts is not None:
                        # Ticket 007: a prescribed split names every
                        # node's per-orbit count outright, so there is
                        # no participation to sample. Sampling here and
                        # discarding the result in ``_split_by_orbit``
                        # would burn entropy, force a `distribution` on
                        # a sequence that needs none, and let a sampled
                        # sequence silently disagree with the
                        # prescription that overrides it.
                        decompositions.append(seq.orbit_counts)
                        continue
                    max_stub_cost = max(seq.orbit_degrees.values())
                    max_per_node = config.degrees // max_stub_cost
                    parts = seq.sample(config.n, rng, max_per_node=max_per_node)
                    decomp = seq._split_by_orbit(parts, rng)
                    decompositions.append(decomp)

                # 2. Allocate subgraphs to nodes (verify degree budget)
                allocation = allocate_subgraphs(
                    degrees=config.degrees,
                    sequences=list(config.sequences),
                    decompositions=decompositions,
                    rng=rng,
                )

                # 3. Connect subgraph instances
                for seq_idx in range(len(config.sequences)):
                    connector.connect_subgraph(
                        sequence=config.sequences[seq_idx],
                        allocation=allocation,
                        sequence_index=seq_idx,
                    )

                # 4. Pair remaining single stubs
                connector.connect_singles(allocation.singles)

                # 5. Assemble into adjacency matrix
                graph = cls(connector.to_csr(), config=config)
                _verify_degrees(graph, config)
                return graph

            except (AllocationError, ConnectionError, ValueError, RuntimeError) as exc:
                # Retry with a fresh random state on any failure
                last_exc = exc
                continue

        msg = f"Failed to generate graph after {config.max_retries} retries"
        raise RuntimeError(msg) from last_exc

    @property
    def clustering_coefficient(self) -> float:
        return global_clustering_coefficient(self._adjacency)

    @property
    def designed_clustering(self) -> float:
        """Designed clustering coefficient implied by the originating config.

        The closed-form value computed from the config before generation
        (see `craeft.graphs.metrics.designed_clustering`), for comparison
        against the realized `clustering_coefficient`. The realized value
        will typically exceed this by a small by-product term (random
        closure from stub pairing plus subgraph overlap).

        Raises:
            ValueError: If the graph was not built via `from_config` (so
                has no attached config to compute the designed value from).
        """
        if self._config is None:
            msg = "designed_clustering requires a graph built via from_config"
            raise ValueError(msg)
        return _designed_clustering(self._config)
