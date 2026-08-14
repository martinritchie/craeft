"""Graph config and class for the configuration model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Self

import numpy as np
from numpy.typing import NDArray

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
            DegreeMismatchError: If config.verify_degrees is True and the
                realized degree sequence differs from config.degrees. Not
                caught by the retry loop above: a mismatch is a correctness
                bug, not a dead-end configuration worth retrying.
        """
        if not config.sequences:
            connector = Connector(config.n, rng)
            connector.connect_singles(config.degrees)
            graph = cls(connector.to_csr())
            _verify_degrees(graph, config)
            return graph

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
                graph = cls(connector.to_csr())
                _verify_degrees(graph, config)
                return graph

            except (AllocationError, ConnectionError, ValueError, RuntimeError) as exc:
                # Retry with a fresh random state on any failure
                last_exc = exc
                continue

        msg = (
            f"Failed to generate graph after {config.max_retries} "
            f"retries"
        )
        raise RuntimeError(msg) from last_exc

    @property
    def clustering_coefficient(self) -> float:
        return global_clustering_coefficient(self._adjacency)
