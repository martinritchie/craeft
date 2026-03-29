"""Degree sequence sampling for the configuration model."""

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
    max_degree = n - 1

    while True:
        degrees = distribution.rvs(size=n, random_state=rng)
        if degrees.max() > max_degree:
            continue
        if degrees.sum() % 2 != 0:
            continue
        return degrees
