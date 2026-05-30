"""Tests for degree sequence sampling."""

import numpy as np
import pytest
from scipy.stats import poisson, randint

from craeft.graphs.configuration_model.sequence import sample_degree_sequence


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
