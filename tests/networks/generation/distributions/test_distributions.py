"""Tests for stub/degree distribution utilities."""

import numpy as np
import pytest

from craeft.networks.generation.distributions import (
    Fixed,
    Poisson,
    prepare_stubs,
)


class TestPrepareStubs:
    """Tests for the prepare_stubs correction logic."""

    def test_sum_is_always_even_adjust(self, rng: np.random.Generator) -> None:
        """Adjust method guarantees even sum."""
        distribution = Fixed(1)
        n = 11  # Odd sum forces correction
        stubs = prepare_stubs(distribution, n, rng, method="adjust")
        assert stubs.sum() % 2 == 0

    @pytest.mark.parametrize("method", ["resample_one", "resample_all"])
    def test_sum_is_always_even_resample(
        self, method: str, rng: np.random.Generator
    ) -> None:
        """Resample methods guarantee even sum."""
        # Must use non-Fixed distribution so resampling can change parity
        distribution = Poisson(mu=3.0)
        n = 50

        stubs = prepare_stubs(distribution, n, rng, method=method)

        assert stubs.sum() % 2 == 0

    def test_adjust_adds_one_when_node_is_zero(self) -> None:
        """When adjust selects a node with 0 stubs, it must add (not subtract)."""
        # Fixed(0) means all nodes have 0 stubs initially
        # With odd n, sum is 0 (even), so no correction needed
        # Use a mix: we need to force selection of a zero node
        distribution = Fixed(0)
        n = 1  # Single node with 0 stubs, sum=0 (even), no correction

        # To test the zero-node branch, we need odd sum with zeros present
        # Create scenario: use Poisson which can produce zeros
        rng = np.random.default_rng(42)
        distribution = Poisson(mu=0.1)  # Low mu → many zeros
        n = 11

        # Run many times to hit the zero-node branch
        for seed in range(100):
            rng = np.random.default_rng(seed)
            stubs = prepare_stubs(distribution, n, rng, method="adjust")
            # Invariant must hold regardless of which branch was taken
            assert stubs.sum() % 2 == 0
            assert np.all(stubs >= 0), "Stubs must remain non-negative"

    def test_adjust_modifies_at_most_one_value(self, rng: np.random.Generator) -> None:
        """Adjust method changes at most one stub count by exactly 1."""
        distribution = Fixed(5)
        n = 11  # Odd sum (55) forces correction

        stubs = prepare_stubs(distribution, n, rng, method="adjust")

        # Count how many values differ from original
        original = np.full(n, 5)
        diff = np.abs(stubs - original)

        assert diff.sum() == 1, "Exactly one value should change by 1"

    def test_resample_one_modifies_values(self, rng: np.random.Generator) -> None:
        """Resample_one corrects odd sums by resampling."""
        distribution = Poisson(mu=3.0)
        n = 50

        stubs = prepare_stubs(distribution, n, rng, method="resample_one")

        assert stubs.sum() % 2 == 0

    def test_reproducibility_with_same_seed(self) -> None:
        """Same seed produces identical results."""
        distribution = Poisson(mu=3.0)
        n = 100

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        stubs1 = prepare_stubs(distribution, n, rng1, method="adjust")
        stubs2 = prepare_stubs(distribution, n, rng2, method="adjust")

        np.testing.assert_array_equal(stubs1, stubs2)


class TestPrepareStubsEdgeCases:
    """Tests for boundary conditions in prepare_stubs."""

    @pytest.mark.parametrize("method", ["adjust", "resample_one", "resample_all"])
    def test_already_even_sum_unchanged(
        self, method: str, rng: np.random.Generator
    ) -> None:
        """When initial sum is even, no correction is applied."""
        distribution = Fixed(4)
        n = 10  # 4 * 10 = 40 (even)

        stubs = prepare_stubs(distribution, n, rng, method=method)

        expected = np.full(n, 4)
        np.testing.assert_array_equal(stubs, expected)

    def test_single_node_adjust(self, rng: np.random.Generator) -> None:
        """Single node with odd stub count gets corrected."""
        distribution = Fixed(3)
        n = 1

        stubs = prepare_stubs(distribution, n, rng, method="adjust")

        assert stubs.sum() % 2 == 0
        assert stubs[0] in (2, 4), "Should be 3 ± 1"

    def test_single_node_zero_stubs(self, rng: np.random.Generator) -> None:
        """Single node with 0 stubs (even sum) needs no correction."""
        distribution = Fixed(0)
        n = 1

        stubs = prepare_stubs(distribution, n, rng, method="adjust")

        assert stubs[0] == 0

    def test_all_zeros_returns_zeros(self, rng: np.random.Generator) -> None:
        """Fixed(0) with any n returns all zeros (sum=0 is even)."""
        distribution = Fixed(0)
        n = 100

        stubs = prepare_stubs(distribution, n, rng, method="adjust")

        np.testing.assert_array_equal(stubs, np.zeros(n, dtype=np.int_))

    @pytest.mark.parametrize("method", ["adjust", "resample_one", "resample_all"])
    def test_large_n_completes(self, method: str, rng: np.random.Generator) -> None:
        """Correction works efficiently with large arrays."""
        distribution = Poisson(mu=5.0)
        n = 10_000

        stubs = prepare_stubs(distribution, n, rng, method=method)

        assert len(stubs) == n
        assert stubs.sum() % 2 == 0
