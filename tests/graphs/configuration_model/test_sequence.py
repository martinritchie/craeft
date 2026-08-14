"""Tests for degree sequence sampling."""

import numpy as np
import pytest
from scipy.stats import poisson, randint

from craeft.graphs.configuration_model.sequence import sample_degree_sequence
from craeft.graphs.configuration_model.sequence.sampling import _sample_sequence


class TestSampleDegreeSequenceValidity:
    """Every sampled sequence must have even sum and bounded degrees."""

    @pytest.mark.parametrize("seed", range(20))
    def test_sum_is_even(self, seed: int) -> None:
        rng = np.random.default_rng(seed)
        degrees = sample_degree_sequence(100, poisson(5), rng)
        assert degrees.sum() % 2 == 0

    @pytest.mark.parametrize("seed", range(20))
    def test_no_degree_exceeds_n_minus_one(self, seed: int) -> None:
        n = 50
        rng = np.random.default_rng(seed)
        degrees = sample_degree_sequence(n, poisson(5), rng)
        assert degrees.max() <= n - 1

    def test_length_matches_n(self) -> None:
        rng = np.random.default_rng(42)
        degrees = sample_degree_sequence(75, poisson(3), rng)
        assert len(degrees) == 75

    def test_all_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        degrees = sample_degree_sequence(100, poisson(5), rng)
        assert np.all(degrees >= 0)


class TestSampleDegreeSequenceReproducibility:
    def test_same_seed_same_result(self) -> None:
        d1 = sample_degree_sequence(100, poisson(5), np.random.default_rng(42))
        d2 = sample_degree_sequence(100, poisson(5), np.random.default_rng(42))
        np.testing.assert_array_equal(d1, d2)

    def test_different_seeds_different_results(self) -> None:
        d1 = sample_degree_sequence(100, poisson(5), np.random.default_rng(1))
        d2 = sample_degree_sequence(100, poisson(5), np.random.default_rng(2))
        assert not np.array_equal(d1, d2)


class TestSampleDegreeSequenceStatistics:
    def test_mean_degree_approximates_distribution_mean(self) -> None:
        """Ensemble mean degree should approximate the distribution mean."""
        lam = 6.0
        means = [
            sample_degree_sequence(
                500, poisson(lam), np.random.default_rng(seed)
            ).mean()
            for seed in range(30)
        ]
        ensemble_mean = np.mean(means)
        assert abs(ensemble_mean - lam) < 0.3

    def test_works_with_different_distributions(self) -> None:
        rng = np.random.default_rng(42)
        # Discrete uniform on [1, 6]
        degrees = sample_degree_sequence(100, randint(1, 7), rng)
        assert degrees.sum() % 2 == 0
        assert degrees.min() >= 1
        assert degrees.max() <= 6


class TestSampleSequencePerNodeCap:
    """_sample_sequence must respect an optional per-node cap (ticket 003).

    Without a per-node cap, participation values are sampled with no
    reference to each node's degree budget, so a build's success or
    failure is down to luck rather than construction. ``max_per_node``
    lets callers pass the remaining degree budget so every sampled
    value is guaranteed to fit.
    """

    def test_sample_respects_per_node_cap(self) -> None:
        rng = np.random.default_rng(0)
        n = 300
        # A distribution with mean well above the caps, so clipping
        # is exercised on (almost) every node.
        caps = rng.integers(0, 5, size=n).astype(np.int_)
        values = _sample_sequence(n, poisson(6), rng, divisor=1, max_per_node=caps)
        assert np.all(values <= caps)

    @pytest.mark.parametrize("seed", range(10))
    def test_sum_stays_divisible_with_cap(self, seed: int) -> None:
        """Divisibility repair must not push any value over its cap."""
        rng = np.random.default_rng(seed)
        n = 200
        caps = rng.integers(1, 6, size=n).astype(np.int_)
        values = _sample_sequence(n, poisson(4), rng, divisor=3, max_per_node=caps)
        assert int(values.sum()) % 3 == 0
        assert np.all(values <= caps)

    def test_cap_of_zero_forces_zero(self) -> None:
        """Nodes with no remaining degree budget get no participation."""
        rng = np.random.default_rng(1)
        n = 50
        caps = np.zeros(n, dtype=np.int_)
        values = _sample_sequence(n, poisson(3), rng, divisor=1, max_per_node=caps)
        assert np.all(values == 0)
