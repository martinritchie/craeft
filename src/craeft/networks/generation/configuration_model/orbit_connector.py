"""Orbit-aware hyperstub connection for incomplete subgraphs.

For motifs where corner types differ (e.g., G7/diamond, G14/bowtie),
nodes must be drawn from typed bins and placed into specific positions.
This module mirrors the connector strategies in connectors.py but
operates on typed bins rather than flat arrays.

Reference:
    Ritchie et al. (2015) "Generation and analysis of networks with a
    prescribed degree sequence and subgraph family", Section 2.3.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from craeft.networks.generation.motifs import Motif

from .connectors import _has_collision


def _build_node_array(
    bins: dict[int, NDArray[np.int_]],
    motif: Motif,
) -> NDArray[np.int_] | None:
    """Draw one node per motif position from typed bins.

    Positions are filled according to the motif's corner_types list.
    For example, G14 with corner_types [1, 2, 2, 1, 1] draws positions
    0, 3, 4 from bins[1] and positions 1, 2 from bins[2].

    Returns None if any bin is empty.
    """
    nodes = np.empty(motif.num_nodes, dtype=np.int_)
    pointers: dict[int, int] = {}

    for pos, ctype in enumerate(motif.corner_types):
        if ctype not in pointers:
            pointers[ctype] = 0
        idx = pointers[ctype]
        if idx >= len(bins.get(ctype, [])):
            return None
        nodes[pos] = bins[ctype][idx]
        pointers[ctype] = idx + 1

    return nodes


@dataclass(frozen=True)
class OrbitConnector(ABC):
    """Base class for orbit-aware connection algorithms."""

    @abstractmethod
    def connect_orbit(
        self,
        bins: dict[int, NDArray[np.int_]],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect typed hyperstub bins to form motif instances.

        Args:
            bins: Mapping from corner type to shuffled array of node IDs.
            motif: The motif structure to instantiate.
            rng: Random number generator.

        Returns:
            Tuple of (rows, cols) edge arrays.
        """


@dataclass(frozen=True)
class OrbitRepeated(OrbitConnector):
    """Resample on collision until valid or exhausted.

    Attributes:
        max_attempts: Maximum consecutive failures before stopping.
    """

    max_attempts: int = 1000

    def connect_orbit(
        self,
        bins: dict[int, NDArray[np.int_]],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect with per-instance resampling on collision."""
        shuffled = {t: rng.permutation(arr) for t, arr in bins.items()}
        all_rows: list[int] = []
        all_cols: list[int] = []
        attempts = 0

        while attempts < self.max_attempts:
            nodes = _build_node_array(shuffled, motif)
            if nodes is None:
                break

            if _has_collision(nodes):
                # Reshuffle remaining bins and retry
                shuffled = {t: rng.permutation(arr) for t, arr in shuffled.items()}
                attempts += 1
                continue

            rows, cols = motif.edges_for(nodes)
            all_rows.extend(rows)
            all_cols.extend(cols)

            # Remove used nodes from bins
            for ctype in motif.type_counts:
                count = motif.type_counts[ctype]
                shuffled[ctype] = shuffled[ctype][count:]
            attempts = 0

        return np.array(all_rows, dtype=np.int_), np.array(all_cols, dtype=np.int_)


@dataclass(frozen=True)
class OrbitRefuse(OrbitConnector):
    """Batch rejection: reshuffles all bins on any collision.

    Attributes:
        max_attempts: Maximum batch rejections.
    """

    max_attempts: int = 100

    def connect_orbit(
        self,
        bins: dict[int, NDArray[np.int_]],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect with full batch rejection on collision."""
        num_instances = _max_instances(bins, motif)
        if num_instances == 0:
            return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

        for _ in range(self.max_attempts):
            shuffled = {t: rng.permutation(arr) for t, arr in bins.items()}
            all_rows: list[int] = []
            all_cols: list[int] = []
            valid = True

            for inst in range(num_instances):
                nodes = _extract_instance(shuffled, motif, inst)
                if _has_collision(nodes):
                    valid = False
                    break
                rows, cols = motif.edges_for(nodes)
                all_rows.extend(rows)
                all_cols.extend(cols)

            if valid:
                return (
                    np.array(all_rows, dtype=np.int_),
                    np.array(all_cols, dtype=np.int_),
                )

        return np.array([], dtype=np.int_), np.array([], dtype=np.int_)


@dataclass(frozen=True)
class OrbitErased(OrbitConnector):
    """Connect without checking, erase invalid edges post-hoc."""

    def connect_orbit(
        self,
        bins: dict[int, NDArray[np.int_]],
        motif: Motif,
        rng: np.random.Generator,
    ) -> tuple[NDArray[np.int_], NDArray[np.int_]]:
        """Connect and remove self-loops and duplicates."""
        num_instances = _max_instances(bins, motif)
        if num_instances == 0:
            return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

        shuffled = {t: rng.permutation(arr) for t, arr in bins.items()}
        edges: set[tuple[int, int]] = set()

        for inst in range(num_instances):
            nodes = _extract_instance(shuffled, motif, inst)
            rows, cols = motif.edges_for(nodes)
            for r, c in zip(rows, cols):
                if r != c:
                    edges.add((min(r, c), max(r, c)))

        if not edges:
            return np.array([], dtype=np.int_), np.array([], dtype=np.int_)

        clean_rows, clean_cols = zip(*edges)
        return np.array(clean_rows, dtype=np.int_), np.array(clean_cols, dtype=np.int_)


def _max_instances(bins: dict[int, NDArray[np.int_]], motif: Motif) -> int:
    """Maximum motif instances formable from the available bins."""
    if not bins:
        return 0
    return min(len(bins.get(t, [])) // motif.type_counts[t] for t in motif.type_counts)


def _extract_instance(
    shuffled: dict[int, NDArray[np.int_]],
    motif: Motif,
    instance: int,
) -> NDArray[np.int_]:
    """Extract node IDs for the i-th motif instance from pre-shuffled bins."""
    nodes = np.empty(motif.num_nodes, dtype=np.int_)
    pointer: dict[int, int] = {}

    for pos, ctype in enumerate(motif.corner_types):
        if ctype not in pointer:
            pointer[ctype] = instance * motif.type_counts[ctype]
        nodes[pos] = shuffled[ctype][pointer[ctype]]
        pointer[ctype] += 1

    return nodes


ConnectorName = str


def get_orbit_connector(name: ConnectorName) -> OrbitConnector:
    """Get an orbit connector by name.

    Args:
        name: One of "repeated", "refuse", or "erased".

    Returns:
        OrbitConnector instance.

    Raises:
        ValueError: If name is not recognized.
    """
    connectors: dict[str, type[OrbitConnector]] = {
        "repeated": OrbitRepeated,
        "refuse": OrbitRefuse,
        "erased": OrbitErased,
    }
    if name not in connectors:
        msg = f"Unknown connector '{name}'. Available: {list(connectors.keys())}"
        raise ValueError(msg)
    return connectors[name]()
