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
) -> NDArray[np.int_]:
    """Sample a sequence with bounded values and divisible sum.

    Rejection samples until all values are in [0, n-1] and the
    sum is divisible by divisor.
    """
    max_value = n - 1

    while True:
        values = distribution.rvs(size=n, random_state=rng)
        if values.min() < 0:
            continue
        if values.max() > max_value:
            continue
        if int(values.sum()) % divisor != 0:
            continue
        return values
