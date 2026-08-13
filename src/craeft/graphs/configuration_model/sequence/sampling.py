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
    max_iterations: int = 10000,
) -> NDArray[np.int_]:
    """Sample a sequence with bounded values and divisible sum.

    Rejection samples until all values are non-negative, no value
    exceeds max_value (when set), and the sum is divisible by
    divisor. Raises after max_iterations rejections.

    Args:
        n: Sequence length.
        distribution: Frozen discrete distribution.
        rng: Random number generator.
        divisor: Target divisor for the sum.
        max_value: Upper bound per value. Defaults to n - 1.
        max_iterations: Maximum attempts before raising.

    Returns:
        Array of n non-negative integers.

    Raises:
        RuntimeError: If max_iterations exhausted.
    """
    if max_value is None:
        max_value = n - 1

    for _ in range(max_iterations):
        values = distribution.rvs(size=n, random_state=rng)
        if values.min() < 0:
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
