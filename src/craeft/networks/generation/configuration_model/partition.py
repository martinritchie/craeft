"""Multinomial partitioning of stubs into corner types and singles.

Implements Step 2 of the CCM algorithm: partitioning allocated stubs into
hyperstub bins for motif formation.

Reference:
    Ritchie et al. (2014) "Higher-order structure and epidemic dynamics
    in clustered networks", Journal of Theoretical Biology 348, 21-32.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from craeft.networks.generation.motifs import Motif


class MixedCardinalityError(ValueError):
    """Raised when a motif has corners with different cardinalities.

    The current implementation assumes uniform-cardinality motifs (e.g.,
    triangles, complete graphs) following Ritchie et al. (2014). Support
    for mixed-cardinality motifs (e.g., G7/diamond, G14/bowtie) requires a more
    sophisticated allocation strategy.
    """


@dataclass(frozen=True)
class StubPartition:
    """Result of partitioning stubs into hyperstub bins.

    Attributes:
        corners: Node IDs for corner hyperstubs. Each node i appears
            `corner_counts[i]` times, ready for connector algorithms.
        singles: Node IDs for single-edge stubs. Each node i appears
            `single_counts[i]` times, for standard configuration model.
        corner_counts: Per-node corner allocation (for diagnostics).
        single_counts: Per-node single allocation (for diagnostics).
    """

    corners: NDArray[np.int_]
    singles: NDArray[np.int_]
    corner_counts: NDArray[np.int_]
    single_counts: NDArray[np.int_]

    @property
    def total_corners(self) -> int:
        """Total corner count across all nodes."""
        return int(self.corner_counts.sum())

    @property
    def total_singles(self) -> int:
        """Total single stub count across all nodes."""
        return int(self.single_counts.sum())


def _validate_uniform_cardinality(motif: Motif) -> int:
    """Validate motif has uniform cardinality and return it.

    The cardinality is the degree of each corner within the motif (how many
    edges it contributes). For triangles, each corner has cardinality 2.

    Args:
        motif: Motif to validate.

    Returns:
        The single cardinality value (corner degree within motif).

    Raises:
        MixedCardinalityError: If motif has corners with different degrees.
    """
    # cardinalities maps corner_type -> degree; we need unique degrees
    cardinality_values = list(motif.cardinalities.values())

    if len(set(cardinality_values)) != 1:
        msg = (
            f"Motif has mixed cardinalities {cardinality_values}. "
            f"Only uniform-cardinality motifs (e.g., triangles, complete graphs) "
            f"are supported. See Ritchie et al. (2014) for the algorithm basis."
        )
        raise MixedCardinalityError(msg)

    return cardinality_values[0]


def partition_stubs(
    stubs: NDArray[np.int_],
    motif: Motif,
    phi: float,
    rng: np.random.Generator,
) -> StubPartition:
    """Multinomially partition stubs into corners and singles.

    For each node with k stubs, probabilistically allocates corners based
    on phi. Each corner consumes `cardinality` stubs; remaining stubs
    become singles for standard configuration model pairing.

    This implements Step 2 of the CCM algorithm from Ritchie et al. (2014),
    which assumes all motif corners have equal cardinality (i.e., fully
    connected subgraphs like triangles or complete graphs).

    Args:
        stubs: Stub count per node (length n). Typically from `prepare_stubs`.
        motif: Motif defining corner structure. Must have uniform cardinality
            (all corners equivalent), e.g., G2 (triangle), G8 (K4), G29 (K5).
        phi: Clustering parameter in [0, 1]. Fraction of maximum possible
            corners to allocate on average. Higher values produce more
            clustered networks.
        rng: Random generator for reproducibility.

    Returns:
        StubPartition containing hyperstub bins ready for connector algorithms.

    Raises:
        MixedCardinalityError: If motif has corners with different cardinalities.
        ValueError: If phi is outside [0, 1].

    Note:
        The assumption of uniform cardinality follows the original CCM paper
        (Ritchie et al., 2014), which uses triangles exclusively. Extending
        to mixed-cardinality motifs requires coordinated allocation to maintain
        corner type ratios.

    Example:
        >>> from craeft.networks.generation.motifs import G2
        >>> from craeft.networks.generation.distributions import (
        ...     prepare_stubs, Poisson,
        ... )
        >>> rng = np.random.default_rng(42)
        >>> stubs = prepare_stubs(Poisson(mu=6.0), n=100, rng=rng)
        >>> partition = partition_stubs(stubs, G2, phi=0.5, rng=rng)
        >>> partition.total_corners  # Number of triangle corners allocated
        150
        >>> partition.total_singles  # Remaining stubs for single edges
        300
    """
    if not 0.0 <= phi <= 1.0:
        msg = f"phi must be in [0, 1], got {phi}"
        raise ValueError(msg)

    cardinality = _validate_uniform_cardinality(motif)

    # Maximum corners each node can participate in
    max_corners = stubs // cardinality

    # Binomial allocation: each potential corner included with probability phi
    corner_counts = rng.binomial(max_corners, phi).astype(np.int_)

    # Remaining stubs become singles
    single_counts = stubs - corner_counts * cardinality

    # Build hyperstub arrays by repeating node IDs
    corners = np.repeat(np.arange(len(stubs)), corner_counts)
    singles = np.repeat(np.arange(len(stubs)), single_counts)

    return StubPartition(
        corners=corners,
        singles=singles,
        corner_counts=corner_counts,
        single_counts=single_counts,
    )
