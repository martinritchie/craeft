"""Configuration model pairing for single stubs.

Implements Step 4 of the CCM algorithm: pairing remaining single stubs
to form simple edges, while avoiding multi-edges with existing motif edges.

Three strategies with different speed/bias tradeoffs mirror the connector
strategies in ccm/connectors.py.

Reference:
    Ritchie et al. (2014) "Higher-order structure and epidemic dynamics
    in clustered networks", Journal of Theoretical Biology 348, 21-32.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


def _canonical_edge(a: int, b: int) -> tuple[int, int]:
    """Return edge in canonical form (smaller, larger)."""
    return (a, b) if a < b else (b, a)


def _is_valid_pair(
    a: int,
    b: int,
    existing: set[tuple[int, int]],
    formed: set[tuple[int, int]],
) -> bool:
    """Check if pairing nodes a and b produces a valid edge.

    A pair is invalid if:
    - It's a self-loop (a == b)
    - It already exists in the motif edges
    - It was already formed in this pairing round
    """
    if a == b:
        return False
    edge = _canonical_edge(a, b)
    return edge not in existing and edge not in formed


@dataclass(frozen=True)
class PairingResult:
    """Result of pairing single stubs.

    Attributes:
        rows: Source node IDs for each edge.
        cols: Target node IDs for each edge.
        unpaired: Number of stubs that couldn't be paired.
    """

    rows: NDArray[np.int_]
    cols: NDArray[np.int_]
    unpaired: int

    @property
    def num_edges(self) -> int:
        """Number of edges formed."""
        return len(self.rows)


@dataclass(frozen=True)
class Pairer(ABC):
    """Base class for single stub pairing algorithms.

    A pairer takes single stubs and existing edges, then pairs stubs
    to form new edges while avoiding self-loops and multi-edges.
    """

    @abstractmethod
    def pair(
        self,
        singles: NDArray[np.int_],
        existing_edges: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> PairingResult:
        """Pair single stubs to form edges.

        Args:
            singles: Array of node IDs, where each node appears once
                per single stub allocated to it.
            existing_edges: Set of edges already formed (from motifs).
                Each edge is a tuple (min_id, max_id) in canonical form.
            rng: Random number generator.

        Returns:
            PairingResult containing formed edges and unpaired count.
        """


@dataclass(frozen=True)
class Repeated(Pairer):
    """Resample individual pairs on collision until valid or exhausted.

    For each consecutive pair in the shuffled stubs, checks validity.
    If invalid, swaps one stub with a random later position and retries.

    Attributes:
        max_attempts: Maximum retries per pair before skipping.
    """

    max_attempts: int = 100

    def pair(
        self,
        singles: NDArray[np.int_],
        existing_edges: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> PairingResult:
        """Pair stubs with per-pair retry on collision."""
        n = len(singles)
        if n < 2:
            return PairingResult(
                rows=np.array([], dtype=np.int_),
                cols=np.array([], dtype=np.int_),
                unpaired=n,
            )

        stubs = singles.copy()
        rng.shuffle(stubs)

        rows: list[int] = []
        cols: list[int] = []
        formed: set[tuple[int, int]] = set()

        i = 0
        while i + 1 < len(stubs):
            a, b = int(stubs[i]), int(stubs[i + 1])

            if _is_valid_pair(a, b, existing_edges, formed):
                edge = _canonical_edge(a, b)
                formed.add(edge)
                rows.append(a)
                cols.append(b)
                i += 2
                continue

            # Try swapping stubs[i+1] with a random later stub
            found = False
            remaining_indices = list(range(i + 2, len(stubs)))
            rng.shuffle(remaining_indices)

            for j in remaining_indices[: self.max_attempts]:
                candidate = int(stubs[j])
                if _is_valid_pair(a, candidate, existing_edges, formed):
                    # Swap and accept
                    stubs[i + 1], stubs[j] = stubs[j], stubs[i + 1]
                    edge = _canonical_edge(a, candidate)
                    formed.add(edge)
                    rows.append(a)
                    cols.append(candidate)
                    found = True
                    break

            if found:
                i += 2
            else:
                # Skip this stub, try pairing it later
                i += 1

        unpaired = n - 2 * len(rows)
        return PairingResult(
            rows=np.array(rows, dtype=np.int_),
            cols=np.array(cols, dtype=np.int_),
            unpaired=unpaired,
        )


@dataclass(frozen=True)
class Refuse(Pairer):
    """Unbiased pairing via batch rejection sampling.

    Shuffles all stubs and pairs consecutively. If any pair is invalid
    (self-loop or multi-edge), rejects the entire batch and reshuffles.
    Guarantees uniform sampling from valid configurations.

    Attributes:
        max_attempts: Maximum batch rejections before giving up.
    """

    max_attempts: int = 1000

    def pair(
        self,
        singles: NDArray[np.int_],
        existing_edges: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> PairingResult:
        """Pair stubs with full batch rejection on any collision."""
        n = len(singles)
        if n < 2:
            return PairingResult(
                rows=np.array([], dtype=np.int_),
                cols=np.array([], dtype=np.int_),
                unpaired=n,
            )

        # Ensure even count
        usable = (n // 2) * 2

        for _ in range(self.max_attempts):
            shuffled = rng.permutation(singles)[:usable]
            rows = shuffled[0::2]
            cols = shuffled[1::2]

            # Check all pairs
            formed: set[tuple[int, int]] = set()
            valid = True

            for a, b in zip(rows, cols):
                a, b = int(a), int(b)
                if not _is_valid_pair(a, b, existing_edges, formed):
                    valid = False
                    break
                formed.add(_canonical_edge(a, b))

            if valid:
                return PairingResult(
                    rows=rows.astype(np.int_),
                    cols=cols.astype(np.int_),
                    unpaired=n - usable,
                )

        # Exhausted attempts
        return PairingResult(
            rows=np.array([], dtype=np.int_),
            cols=np.array([], dtype=np.int_),
            unpaired=n,
        )


@dataclass(frozen=True)
class Erased(Pairer):
    """Fastest pairing with post-hoc edge removal.

    Shuffles stubs and pairs consecutively without collision checking.
    Then removes self-loops, multi-edges, and duplicates with existing edges.
    Fastest approach but produces biased output (high-degree nodes lose edges).
    """

    def pair(
        self,
        singles: NDArray[np.int_],
        existing_edges: set[tuple[int, int]],
        rng: np.random.Generator,
    ) -> PairingResult:
        """Pair stubs and erase invalid edges post-hoc."""
        n = len(singles)
        if n < 2:
            return PairingResult(
                rows=np.array([], dtype=np.int_),
                cols=np.array([], dtype=np.int_),
                unpaired=n,
            )

        # Ensure even count
        usable = (n // 2) * 2

        shuffled = rng.permutation(singles)[:usable]
        raw_rows = shuffled[0::2]
        raw_cols = shuffled[1::2]

        # Filter to valid edges
        edges: set[tuple[int, int]] = set()
        for a, b in zip(raw_rows, raw_cols):
            a, b = int(a), int(b)
            if a == b:
                continue
            edge = _canonical_edge(a, b)
            if edge in existing_edges:
                continue
            edges.add(edge)

        if not edges:
            return PairingResult(
                rows=np.array([], dtype=np.int_),
                cols=np.array([], dtype=np.int_),
                unpaired=n,
            )

        clean_rows, clean_cols = zip(*edges)
        formed = len(edges)
        unpaired = n - 2 * formed

        return PairingResult(
            rows=np.array(clean_rows, dtype=np.int_),
            cols=np.array(clean_cols, dtype=np.int_),
            unpaired=unpaired,
        )


PairerName = Literal["repeated", "refuse", "erased"]


def get_pairer(name: PairerName) -> Pairer:
    """Get a pairer by name.

    Args:
        name: One of "repeated", "refuse", or "erased".

    Returns:
        Pairer instance with default configuration.

    Raises:
        ValueError: If name is not recognized.
    """
    pairers: dict[str, type[Pairer]] = {
        "repeated": Repeated,
        "refuse": Refuse,
        "erased": Erased,
    }
    if name not in pairers:
        msg = f"Unknown pairer '{name}'. Available: {list(pairers.keys())}"
        raise ValueError(msg)
    return pairers[name]()


def pair_singles(
    singles: NDArray[np.int_],
    existing_edges: set[tuple[int, int]],
    rng: np.random.Generator,
    strategy: PairerName = "repeated",
) -> PairingResult:
    """Pair single stubs to form edges using the specified strategy.

    Convenience function that selects and applies a pairing strategy.

    Args:
        singles: Array of node IDs (from StubPartition.singles).
        existing_edges: Edges already formed from motifs. Each edge should
            be in canonical form (min_id, max_id).
        rng: Random number generator.
        strategy: Pairing strategy - "repeated", "refuse", or "erased".

    Returns:
        PairingResult with formed edges and unpaired stub count.

    Example:
        >>> from craeft.networks.generation.configuration_model.partition import (
        ...     partition_stubs,
        ... )
        >>> from craeft.networks.generation.motifs import G2, Repeated
        >>> rng = np.random.default_rng(42)
        >>> # ... after partition and motif connection ...
        >>> existing = {(0, 1), (1, 2), (0, 2)}  # triangle edges
        >>> singles = np.array([0, 1, 3, 3, 4, 4])
        >>> result = pair_singles(singles, existing, rng, strategy="repeated")
        >>> result.num_edges
        3
    """
    pairer = get_pairer(strategy)
    return pairer.pair(singles, existing_edges, rng)
