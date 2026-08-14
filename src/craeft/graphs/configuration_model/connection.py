"""Edge formation for the configuration model family.

Stateful connector that accumulates edges from subgraph connection
and single stub pairing, tracking existing edges to prevent
multi-edges across steps.

Reference:
    Ritchie et al. (2016), Algorithm 1 (p.278), lines 16-27.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import coo_matrix, csr_matrix

if TYPE_CHECKING:
    from craeft.graphs.configuration_model.sequence import (
        Allocation,
        SubgraphSequence,
    )


class Connector:
    """Assembles edges from subgraph connection and single pairing.

    Tracks formed edges across subgraph types to prevent multi-edges
    during the connection process. Used by both vanilla CM (singles
    only) and CMA (subgraphs then singles).
    """

    def __init__(self, n: int, rng: np.random.Generator) -> None:
        self._n = n
        self._rng = rng
        self._rows: list[int] = []
        self._cols: list[int] = []
        self._existing: set[tuple[int, int]] = set()

    def connect_subgraph(
        self,
        sequence: SubgraphSequence,
        allocation: Allocation,
        sequence_index: int,
        max_attempts: int = 1000,
    ) -> None:
        """Form edges for one subgraph from its allocated bins.

        For each vertex position in the subgraph, determines its
        orbit and draws a concrete node from that orbit's pool.
        Shuffles pools independently, then pops vertices in order
        to form each instance. Rejects groups with duplicate nodes
        or existing edges.

        Accumulates edges and updates the existing-edges set.

        Args:
            sequence: The subgraph sequence being connected.
            allocation: Subgraph allocation with per-orbit node bins.
            sequence_index: Index of this sequence in the allocation.
            max_attempts: Maximum consecutive failures before raising.

        Raises:
            ConnectionError: If max_attempts exceeded.
        """
        orbit_ids = sorted(allocation.bins)
        this_bin_keys = [
            k for k in orbit_ids if k[0] == sequence_index
        ]

        if not this_bin_keys:
            return  # no bins for this sequence

        # Collect per-orbit node pools
        orbit_pools: dict[int, list[int]] = {}
        for key in this_bin_keys:
            _, orbit = key
            orbit_pools[orbit] = allocation.node_ids_for(*key).tolist()

        # Compute number of instances from any non-empty pool
        num_instances = next(
            (len(pool) // sequence.orbit_sizes[o]
             for o, pool in orbit_pools.items() if pool),
            0,
        )

        if num_instances == 0:
            return

        # Verify each orbit pool has exactly sizes[o] * num_instances nodes
        sizes = sequence.orbit_sizes
        for o, pool in orbit_pools.items():
            expected = sizes[o] * num_instances
            if len(pool) != expected:
                msg = (
                    f"Orbit {o} pool size {len(pool)} != "
                    f"sizes[{o}]({sizes[o]}) * {num_instances}"
                )
                raise ConnectionError(msg)

        # Vertex → orbit mapping
        vertex_orbit = sequence.orbits  # list: orbit label per vertex

        attempts = 0
        while attempts < max_attempts:
            # Shuffle each orbit pool independently
            for pool in orbit_pools.values():
                if pool:
                    self._rng.shuffle(pool)

            # Copy pools so we can restore on failure
            working_pools = {
                o: list(pool) for o, pool in orbit_pools.items()
            }

            collision = False
            # Edges proposed by earlier instances within this same
            # attempt aren't in self._existing yet (that's only
            # updated on commit) — without tracking them separately,
            # two instances of a subgraph within one attempt could
            # silently reuse the same edges (e.g. two triangle
            # instances drawing the same 3 nodes), producing
            # multi-edges instead of the intended collision/retry.
            proposed: set[tuple[int, int]] = set()
            for _ in range(num_instances):
                group = np.empty(len(vertex_orbit), dtype=np.int_)
                for v, orb in enumerate(vertex_orbit):
                    group[v] = working_pools[orb].pop()

                # Check for duplicate nodes within the group
                if len(set(group.tolist())) < len(group):
                    collision = True
                    break

                # Check for existing edges (from prior connector
                # calls, or from earlier instances this attempt)
                rows, cols = sequence.edges_for(group)
                instance_edges = [
                    (min(r, c), max(r, c)) for r, c in zip(rows, cols)
                ]
                if any(
                    edge in self._existing or edge in proposed
                    for edge in instance_edges
                ):
                    collision = True
                    break
                proposed.update(instance_edges)

            if not collision:
                # Commit: record edges from the shuffled pools
                for _ in range(num_instances):
                    group = np.empty(len(vertex_orbit), dtype=np.int_)
                    for v, orb in enumerate(vertex_orbit):
                        group[v] = orbit_pools[orb].pop()
                    rows, cols = sequence.edges_for(group)
                    for r, c in zip(rows, cols):
                        edge = (min(r, c), max(r, c))
                        self._existing.add(edge)
                        self._rows.append(r)
                        self._cols.append(c)
                break

            attempts += 1

        if attempts >= max_attempts:
            raise ConnectionError(
                f"Failed to connect subgraph after {max_attempts} "
                f"consecutive collisions"
            )

    def connect_singles(self, singles: NDArray[np.int_]) -> None:
        """Pair remaining single stubs.

        Shuffles and pairs consecutive stubs, avoiding self-loops,
        multi-edges, and edges already formed by subgraph connection.

        Args:
            singles: Per-node stub counts. Sum must be even.

        Raises:
            ValueError: If stub sum is odd.
        """
        total = int(singles.sum())

        if total % 2 != 0:
            msg = f"Stub sum must be even, got {total}"
            raise ValueError(msg)

        if total == 0:
            return

        stubs = np.repeat(np.arange(len(singles)), singles)
        self._rng.shuffle(stubs)

        rows = stubs[0::2]
        cols = stubs[1::2]

        for r, c in zip(rows.tolist(), cols.tolist()):
            if r == c:
                continue
            edge = (min(r, c), max(r, c))
            if edge in self._existing:
                continue
            self._existing.add(edge)
            self._rows.append(r)
            self._cols.append(c)

    def to_csr(self) -> csr_matrix:
        """Assemble all accumulated edges into a symmetric adjacency matrix."""
        if not self._rows:
            return csr_matrix((self._n, self._n), dtype=np.int8)

        rows = np.array(self._rows, dtype=np.int_)
        cols = np.array(self._cols, dtype=np.int_)

        sym_rows = np.concatenate([rows, cols])
        sym_cols = np.concatenate([cols, rows])
        data = np.ones(len(sym_rows), dtype=np.int8)

        return coo_matrix(
            (data, (sym_rows, sym_cols)),
            shape=(self._n, self._n),
            dtype=np.int8,
        ).tocsr()


class ConnectionError(Exception):
    """Raised when the connection process cannot form valid subgraph instances."""
