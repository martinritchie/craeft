"""Hyperstub connection algorithms.

Three strategies with different speed/bias tradeoffs for forming motif
instances from allocated hyperstubs.

Reference:
    Ritchie et al. (2015): "Generation and analysis of networks with a
    prescribed degree sequence and subgraph family"
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from craeft.networks.generation.motifs import Motif


def _has_collision(nodes: NDArray[np.int_]) -> bool:
    """Check if any node appears more than once."""
    return len(set(nodes.tolist())) < len(nodes)


@dataclass(frozen=True)
class Connector(ABC):
    """Base class for hyperstub connection algorithms.

    A connector takes bins of hyperstubs (nodes allocated to participate
    in motif instances) and forms the actual edges by matching hyperstubs
    into complete motif instances.
    """

    @abstractmethod
    def connect(
        self,
        hyperstubs: NDArray[np.int_],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect hyperstubs to form motif instances.

        Args:
            hyperstubs: Array of node IDs, where each node appears once
                per hyperstub allocated to it. Length must be divisible
                by motif.num_nodes.
            motif: The motif structure to instantiate.
            rng: Random number generator.

        Returns:
            Tuple of (rows, cols) arrays representing edges. Each motif
            instance contributes motif.num_edges edges.
        """


@dataclass(frozen=True)
class Repeated(Connector):
    """Resample on collision until valid or exhausted.

    Samples k hyperstubs at a time. If any node appears twice in the sample
    (a collision), discards and resamples. On success, removes those hyperstubs
    and continues until fewer than k remain.

    Attributes:
        max_attempts: Maximum consecutive failures before stopping.
    """

    max_attempts: int = 1000

    def connect(
        self,
        hyperstubs: NDArray[np.int_],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect hyperstubs with resampling on collision."""
        k = motif.num_nodes
        available = hyperstubs.copy()

        all_rows: list[int] = []
        all_cols: list[int] = []
        attempts = 0

        while len(available) >= k and attempts < self.max_attempts:
            # Sample k positions from what remains
            positions = rng.choice(len(available), size=k, replace=False)
            candidate = available[positions]

            if _has_collision(candidate):
                attempts += 1
                continue

            # Valid: extract edges and remove used hyperstubs
            rows, cols = motif.edges_for(candidate)
            all_rows.extend(rows)
            all_cols.extend(cols)

            available = np.delete(available, positions)
            attempts = 0

        return np.array(all_rows, dtype=np.int_), np.array(all_cols, dtype=np.int_)


@dataclass(frozen=True)
class Refuse(Connector):
    """Unbiased connection via rejection sampling.

    Shuffles all hyperstubs and partitions into groups of k. If any group
    contains a collision (same node twice), rejects the entire batch and
    reshuffles. Guarantees uniform sampling from valid configurations.

    Attributes:
        max_attempts: Maximum batch rejections before giving up.
    """

    max_attempts: int = 100

    def connect(
        self,
        hyperstubs: NDArray[np.int_],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect hyperstubs with full batch rejection on collision."""
        k = motif.num_nodes
        n = len(hyperstubs)

        if n < k:
            return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

        num_motifs = n // k
        usable = num_motifs * k

        for _ in range(self.max_attempts):
            # Shuffle and partition into groups of k
            shuffled = rng.permutation(hyperstubs)[:usable]
            groups = shuffled.reshape(num_motifs, k)

            # Reject if any group has a collision
            if any(_has_collision(group) for group in groups):
                continue

            # All groups valid: extract edges
            all_rows: list[int] = []
            all_cols: list[int] = []
            for group in groups:
                rows, cols = motif.edges_for(group)
                all_rows.extend(rows)
                all_cols.extend(cols)

            return np.array(all_rows, dtype=np.int_), np.array(all_cols, dtype=np.int_)

        # Exhausted attempts: return empty
        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)


@dataclass(frozen=True)
class Erased(Connector):
    """Fastest connection with post-hoc edge removal.

    Shuffles hyperstubs and partitions into groups without collision checking.
    Forms all motif instances, then removes self-loops and duplicate edges.
    Fastest approach but produces biased output (high-degree nodes lose edges).

    Use when speed is critical and slight bias is acceptable.
    """

    def connect(
        self,
        hyperstubs: NDArray[np.int_],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect hyperstubs and erase invalid edges post-hoc."""
        k = motif.num_nodes
        n = len(hyperstubs)

        if n < k:
            return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

        num_motifs = n // k
        usable = num_motifs * k

        # Shuffle and partition into groups (ignore collisions)
        shuffled = rng.permutation(hyperstubs)[:usable]
        groups = shuffled.reshape(num_motifs, k)

        # Extract all edges
        all_rows: list[int] = []
        all_cols: list[int] = []
        for group in groups:
            rows, cols = motif.edges_for(group)
            all_rows.extend(rows)
            all_cols.extend(cols)

        # Remove self-loops and duplicates
        edges: set[tuple[int, int]] = set()
        for r, c in zip(all_rows, all_cols):
            if r != c:  # Skip self-loops
                edge = (min(r, c), max(r, c))  # Canonical form
                edges.add(edge)

        if not edges:
            return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

        clean_rows, clean_cols = zip(*edges)
        return np.array(clean_rows, dtype=np.int_), np.array(clean_cols, dtype=np.int_)


ConnectorName = Literal["repeated", "refuse", "erased"]


def get_connector(name: ConnectorName) -> Connector:
    """Get a connector by name.

    Args:
        name: One of "repeated", "refuse", or "erased".

    Returns:
        Connector instance with default configuration.

    Raises:
        ValueError: If name is not recognized.
    """
    connectors: dict[str, type[Connector]] = {
        "repeated": Repeated,
        "refuse": Refuse,
        "erased": Erased,
    }
    if name not in connectors:
        msg = f"Unknown connector '{name}'. Available: {list(connectors.keys())}"
        raise ValueError(msg)
    return connectors[name]()
