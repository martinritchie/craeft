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

        Populates typed bins, shuffles, pops groups of nodes,
        checks for duplicates and existing edges. Reshuffles
        on collision. Accumulates edges and updates existing set.

        Args:
            sequence: The subgraph sequence being connected.
            allocation: Subgraph allocation from greedy assignment.
            sequence_index: Index of this sequence in the allocation.
            max_attempts: Maximum consecutive failures before raising.

        Raises:
            ConnectionError: If max_attempts exceeded.
        """
        ...

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
