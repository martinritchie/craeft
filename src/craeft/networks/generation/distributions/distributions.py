"""Stub/degree distributions for network generation."""

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray


class Distribution(Protocol):
    """Protocol for discrete distributions used in stub allocation."""

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.int_]:
        """Sample n values from the distribution.

        Args:
            n: Number of values to sample.
            rng: Random number generator.

        Returns:
            Array of n non-negative integers.
        """
        ...


@dataclass(frozen=True)
class Poisson:
    """Poisson distribution with mean mu."""

    mu: float

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.int_]:
        """Sample n values from Poisson(mu)."""
        return rng.poisson(self.mu, size=n)


@dataclass(frozen=True)
class Fixed:
    """Fixed value (all nodes get the same stub count)."""

    value: int

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.int_]:
        """Return array of n identical values."""
        return np.full(n, self.value, dtype=np.int_)


@dataclass(frozen=True)
class Empirical:
    """Empirical distribution from observed values."""

    values: tuple[int, ...]

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.int_]:
        """Sample n values with replacement from observed values."""
        return rng.choice(self.values, size=n, replace=True)


CorrectionMethod = Literal["adjust", "resample_one", "resample_all"]


def prepare_stubs(
    distribution: Distribution,
    n: int,
    rng: np.random.Generator,
    method: CorrectionMethod = "resample_one",
) -> NDArray[np.int_]:
    """Sample stubs from distribution and ensure even sum.

    Args:
        distribution: Distribution to sample from.
        n: Number of nodes.
        rng: Random number generator.
        method: How to correct odd sums:
            - "adjust": Add or subtract 1 from a random node
            - "resample_one": Resample one node until sum is even
            - "resample_all": Resample all nodes until sum is even

    Returns:
        Array of stub counts with even sum.
    """
    stubs = distribution.sample(n, rng)

    if method == "resample_all":
        while stubs.sum() % 2 == 1:
            stubs = distribution.sample(n, rng)

    elif method == "resample_one":
        while stubs.sum() % 2 == 1:
            i = rng.integers(n)
            stubs[i] = distribution.sample(1, rng)[0]

    elif method == "adjust":
        if stubs.sum() % 2 == 1:
            i = rng.integers(n)
            if stubs[i] == 0:
                stubs[i] += 1
            else:
                stubs[i] += rng.choice(np.array([-1, 1]))

    return stubs
