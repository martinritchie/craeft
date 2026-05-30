"""Multinomial decomposition of subgraph sequences into corner-type counts.

For each node, splits its subgraph participation count into counts per
corner type using the multinomial distribution. Ensures totals satisfy
the handshake lemma (divisibility by group size).

Reference:
    Ritchie et al. (2015) "Generation and analysis of networks with a
    prescribed degree sequence and subgraph family", Section 2.2.
"""

import numpy as np
from numpy.typing import NDArray

from craeft.networks.generation.motifs import Motif

from .subgraph_spec import SubgraphSpec

_MAX_RESAMPLE_ATTEMPTS = 10_000


def multinomial_decompose(
    spec: SubgraphSpec,
    rng: np.random.Generator,
) -> dict[int, NDArray[np.int_]]:
    """Decompose subgraph sequence into per-corner-type counts.

    For complete motifs (all corners equivalent), returns a single entry.
    For incomplete motifs, uses the multinomial distribution to split
    each node's count across corner types proportionally.

    Args:
        spec: Subgraph specification with motif and per-node sequence.
        rng: Random generator for multinomial sampling.

    Returns:
        Mapping from corner type to per-node count array.

    Raises:
        RuntimeError: If handshake balancing fails after max attempts.
    """
    motif = spec.motif

    if motif.is_complete:
        return _decompose_complete(spec)

    return _decompose_incomplete(spec, rng)


def _decompose_complete(spec: SubgraphSpec) -> dict[int, NDArray[np.int_]]:
    """Trivial decomposition for complete (uniform-cardinality) motifs."""
    corner_type = spec.motif.corner_types[0]
    return {corner_type: spec.sequence.copy()}


def _decompose_incomplete(
    spec: SubgraphSpec,
    rng: np.random.Generator,
) -> dict[int, NDArray[np.int_]]:
    """Multinomial decomposition for incomplete (mixed-cardinality) motifs.

    Draws from Multinomial(sequence[i], p) per node, where p reflects
    the proportion of each corner type in the motif. Resamples until
    each type total is divisible by the number of corners of that type.
    """
    motif = spec.motif
    types = sorted(motif.type_counts.keys())
    proportions = _compute_proportions(motif, types)

    for _ in range(_MAX_RESAMPLE_ATTEMPTS):
        decomposition = _sample_once(spec.sequence, types, proportions, rng)
        if _is_balanced(decomposition, motif):
            return decomposition

    msg = (
        f"Failed to balance corner types after {_MAX_RESAMPLE_ATTEMPTS} "
        f"attempts for motif with types {types}"
    )
    raise RuntimeError(msg)


def _compute_proportions(
    motif: Motif,
    types: list[int],
) -> NDArray[np.float64]:
    """Compute multinomial proportions from motif type counts."""
    counts = np.array([motif.type_counts[t] for t in types], dtype=np.float64)
    return counts / counts.sum()


def _sample_once(
    sequence: NDArray[np.int_],
    types: list[int],
    proportions: NDArray[np.float64],
    rng: np.random.Generator,
) -> dict[int, NDArray[np.int_]]:
    """Single multinomial draw across all nodes."""
    n = len(sequence)
    result = {t: np.zeros(n, dtype=np.int_) for t in types}

    for i in range(n):
        if sequence[i] == 0:
            continue
        draws = rng.multinomial(int(sequence[i]), proportions)
        for j, t in enumerate(types):
            result[t][i] = draws[j]

    return result


def _is_balanced(
    decomposition: dict[int, NDArray[np.int_]],
    motif: Motif,
) -> bool:
    """Check handshake lemma: each type total divisible by its count."""
    for corner_type, counts in decomposition.items():
        total = int(counts.sum())
        group_size = motif.type_counts[corner_type]
        if total % group_size != 0:
            return False
    return True


def compute_stub_cost(
    decomposition: dict[int, NDArray[np.int_]],
    motif: Motif,
) -> NDArray[np.int_]:
    """Compute per-node stub cost from a corner-type decomposition.

    Each corner of type t consumes cardinality[t] stubs. The total cost
    per node is the sum across all corner types.

    Args:
        decomposition: Corner-type counts from multinomial_decompose().
        motif: The motif defining cardinalities.

    Returns:
        Per-node array of stubs consumed by this subgraph's allocation.
    """
    n = len(next(iter(decomposition.values())))
    cost = np.zeros(n, dtype=np.int_)
    for corner_type, counts in decomposition.items():
        cost += counts * motif.cardinalities[corner_type]
    return cost
