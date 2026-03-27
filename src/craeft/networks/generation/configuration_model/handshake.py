"""Handshake lemma balancing for hyperstub allocations.

After cardinality matching, ensures that each corner type's total count
is divisible by the required group size and that singles have even total.
Sheds excess back to singles where needed.
"""

import numpy as np

from .hyperstub_matrix import HyperstubAllocation


def balance_allocation(allocation: HyperstubAllocation) -> HyperstubAllocation:
    """Balance a hyperstub allocation to satisfy the handshake lemma.

    For each (spec, corner_type) bin, trims the total count down to the
    nearest multiple of the required group size. Trimmed hyperstubs are
    returned to the singles pool. Also ensures the singles total is even.

    Args:
        allocation: Unbalanced allocation from cardinality matching.

    Returns:
        A new HyperstubAllocation with balanced bins and adjusted singles.
    """
    bins = {k: v.copy() for k, v in allocation.bins.items()}
    singles = allocation.singles.copy()
    specs = allocation.specs

    for (spec_idx, corner_type), counts in bins.items():
        motif = specs[spec_idx].motif
        group_size = motif.type_counts[corner_type]
        total = int(counts.sum())
        excess = total % group_size

        if excess > 0:
            _shed_excess(counts, excess, singles, motif.cardinalities[corner_type])

    # Ensure even singles total
    total_singles = int(singles.sum())
    if total_singles % 2 != 0:
        # Find a node with at least one single and remove it
        candidates = np.where(singles > 0)[0]
        if len(candidates) > 0:
            singles[candidates[0]] -= 1

    return HyperstubAllocation(
        bins=bins,
        singles=singles,
        specs=specs,
    )


def _shed_excess(
    counts: np.ndarray,
    excess: int,
    singles: np.ndarray,
    cardinality: int,
) -> None:
    """Remove excess hyperstubs from a bin, returning stubs to singles."""
    remaining = excess
    for i in range(len(counts)):
        if remaining <= 0:
            break
        shed = min(int(counts[i]), remaining)
        if shed > 0:
            counts[i] -= shed
            singles[i] += shed * cardinality
            remaining -= shed
