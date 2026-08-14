"""Rejection sampling for degree and participation sequences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_discrete


def sample_degree_sequence(
    n: int,
    distribution: rv_discrete,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Sample a valid degree sequence from a discrete distribution.

    Draws n degrees independently and rejects sequences where the
    sum is odd or any degree exceeds n - 1. Expected few attempts
    for well-chosen distributions.

    Args:
        n: Number of nodes.
        distribution: Frozen scipy discrete distribution (e.g. poisson(3)).
        rng: Random number generator.

    Returns:
        Array of n non-negative degrees with even sum,
        each at most n - 1.
    """
    return _sample_sequence(n, distribution, rng, divisor=2)


def _sample_sequence(
    n: int,
    distribution: rv_discrete,
    rng: np.random.Generator,
    divisor: int,
    max_value: int | None = None,
    max_per_node: NDArray[np.int_] | None = None,
    max_iterations: int = 10000,
) -> NDArray[np.int_]:
    """Sample a sequence with bounded values and divisible sum.

    Without ``max_per_node``: rejection samples until all values are
    non-negative, no value exceeds max_value (when set), and the sum
    is divisible by divisor. Raises after max_iterations rejections.

    With ``max_per_node``: rejection sampling against a tight,
    node-specific cap can fail almost surely (see ticket 003 — with
    comparable spread between the participation and degree
    distributions, rejecting until every node is simultaneously
    under budget essentially never terminates). Instead, values are
    *clipped* to their per-node cap, then the sum is repaired to stay
    divisible by ``divisor`` by nudging nodes that still have slack
    under their cap upward (falling back to nudging positive-valued
    nodes downward if there isn't enough slack). This is deterministic
    truncation, not rejection: it biases the realized sequence's mean
    and variance downward relative to ``distribution`` — the cap
    exists precisely to prevent values that would blow the caller's
    budget, so the bias is the deliberate, documented cost of
    guaranteeing every value fits. Callers who need distributional
    purity should pass a prescribed sequence directly instead of
    sampling.

    Args:
        n: Sequence length.
        distribution: Frozen discrete distribution.
        rng: Random number generator.
        divisor: Target divisor for the sum.
        max_value: Upper bound per value. Defaults to n - 1.
        max_per_node: Optional length-n array of per-node upper
            bounds (e.g. a remaining degree budget), applied in
            addition to and no looser than ``max_value``. When given,
            over-cap values are clipped rather than rejected.
        max_iterations: Maximum attempts before raising.

    Returns:
        Array of n non-negative integers, each at most its
        corresponding bound, whose sum is divisible by divisor.

    Raises:
        RuntimeError: If max_iterations exhausted.
    """
    if max_value is None:
        max_value = n - 1

    caps: NDArray[np.int_] | None = None
    if max_per_node is not None:
        caps = np.minimum(np.asarray(max_per_node), max_value)

    for _ in range(max_iterations):
        values = distribution.rvs(size=n, random_state=rng)
        if values.min() < 0:
            continue

        if caps is not None:
            values = np.minimum(values, caps)
            if int(values.sum()) % divisor != 0:
                values = _fix_divisibility(values, caps, divisor, rng)
            if int(values.sum()) % divisor == 0:
                return values
            continue

        if values.max() > max_value:
            continue
        if int(values.sum()) % divisor != 0:
            continue
        return values

    msg = (
        f"Failed to sample valid sequence after {max_iterations} "
        f"attempts (n={n}, max_value={max_value}, divisor={divisor})"
    )
    raise RuntimeError(msg)


def _fix_divisibility(
    values: NDArray[np.int_],
    caps: NDArray[np.int_],
    divisor: int,
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    """Nudge values by +/-1 so their sum becomes divisible by divisor.

    Prefers raising nodes that still have slack (value < cap) to avoid
    compounding the downward bias already introduced by clipping in
    ``_sample_sequence``. Falls back to lowering positive-valued nodes
    if there isn't enough slack to reach the next multiple of
    ``divisor`` upward (e.g. every node is already at its cap).

    Args:
        values: Already-clipped sequence (not mutated in place).
        caps: Per-node upper bound, same shape as values.
        divisor: Target divisor for the sum.
        rng: Random number generator, used to pick which nodes absorb
            the adjustment so no node position is systematically
            favoured.

    Returns:
        A new array with the same or adjacent sum, divisible by
        divisor when enough slack or positive mass was available.
    """
    values = values.copy()
    remainder = int(values.sum()) % divisor
    if remainder == 0:
        return values

    needed = divisor - remainder
    slack = caps - values
    donors = np.where(slack > 0)[0]
    rng.shuffle(donors)
    for idx in donors:
        if needed == 0:
            break
        bump = min(needed, int(slack[idx]))
        values[idx] += bump
        needed -= bump

    if needed == 0:
        return values

    remainder = int(values.sum()) % divisor
    if remainder == 0:
        return values
    positive = np.where(values > 0)[0]
    rng.shuffle(positive)
    for idx in positive:
        if remainder == 0:
            break
        drop = min(remainder, int(values[idx]))
        values[idx] -= drop
        remainder -= drop

    return values
