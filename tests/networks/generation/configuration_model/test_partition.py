"""Tests for multinomial stub partitioning."""

import numpy as np
import pytest

from craeft.networks.generation.configuration_model.partition import (
    MixedCardinalityError,
    partition_stubs,
)
from craeft.networks.generation.motifs import G2, G7, G8


class TestPartitionStubsInvariants:
    """Tests for core invariants that must always hold."""

    def test_stubs_are_conserved(self, rng: np.random.Generator) -> None:
        """Every stub becomes either a corner contribution or a single."""
        stubs = np.array([6, 8, 4, 10, 2])
        cardinality = 2  # G2 corners consume 2 stubs each

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        # For each node: corners * cardinality + singles == original stubs
        reconstructed = result.corner_counts * cardinality + result.single_counts
        np.testing.assert_array_equal(reconstructed, stubs)

    def test_all_counts_non_negative(self, rng: np.random.Generator) -> None:
        """Corner and single counts must never be negative."""
        stubs = np.array([6, 8, 4, 10, 2, 0])

        result = partition_stubs(stubs, G2, phi=0.7, rng=rng)

        assert np.all(result.corner_counts >= 0)
        assert np.all(result.single_counts >= 0)

    def test_corners_array_length_matches_sum(self, rng: np.random.Generator) -> None:
        """The corners array has one entry per allocated corner."""
        stubs = np.array([6, 8, 4, 10])

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert len(result.corners) == result.corner_counts.sum()

    def test_singles_array_length_matches_sum(self, rng: np.random.Generator) -> None:
        """The singles array has one entry per single stub."""
        stubs = np.array([6, 8, 4, 10])

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert len(result.singles) == result.single_counts.sum()

    def test_node_ids_in_corners_are_valid(self, rng: np.random.Generator) -> None:
        """Corner array contains only valid node indices."""
        stubs = np.array([6, 8, 4, 10])
        n = len(stubs)

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert np.all(result.corners >= 0)
        assert np.all(result.corners < n)

    def test_node_ids_in_singles_are_valid(self, rng: np.random.Generator) -> None:
        """Singles array contains only valid node indices."""
        stubs = np.array([6, 8, 4, 10])
        n = len(stubs)

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert np.all(result.singles >= 0)
        assert np.all(result.singles < n)


class TestPartitionStubsBoundaries:
    """Tests for boundary conditions and edge cases."""

    def test_phi_zero_allocates_no_corners(self, rng: np.random.Generator) -> None:
        """With phi=0, all stubs become singles."""
        stubs = np.array([6, 8, 4, 10])

        result = partition_stubs(stubs, G2, phi=0.0, rng=rng)

        np.testing.assert_array_equal(result.corner_counts, np.zeros(4))
        np.testing.assert_array_equal(result.single_counts, stubs)

    def test_phi_one_allocates_max_corners(self, rng: np.random.Generator) -> None:
        """With phi=1, each node gets maximum possible corners."""
        stubs = np.array([6, 8, 4, 10])
        cardinality = 2

        result = partition_stubs(stubs, G2, phi=1.0, rng=rng)

        expected_corners = stubs // cardinality
        np.testing.assert_array_equal(result.corner_counts, expected_corners)

        # Singles are the remainder
        expected_singles = stubs % cardinality
        np.testing.assert_array_equal(result.single_counts, expected_singles)

    def test_node_with_insufficient_stubs_gets_no_corners(
        self, rng: np.random.Generator
    ) -> None:
        """Node with fewer stubs than cardinality cannot form corners."""
        # G8 has cardinality 3 (each corner connects to 3 others)
        stubs = np.array([2, 6, 1, 9])  # Nodes 0 and 2 have < 3 stubs

        result = partition_stubs(stubs, G8, phi=1.0, rng=rng)

        # Nodes with < 3 stubs get 0 corners even with phi=1
        assert result.corner_counts[0] == 0
        assert result.corner_counts[2] == 0

    def test_all_zero_stubs(self, rng: np.random.Generator) -> None:
        """Nodes with zero stubs produce empty partition."""
        stubs = np.array([0, 0, 0, 0])

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert len(result.corners) == 0
        assert len(result.singles) == 0

    def test_single_node(self, rng: np.random.Generator) -> None:
        """Works with single-node input."""
        stubs = np.array([6])

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert len(result.corner_counts) == 1
        assert len(result.single_counts) == 1
        # Conservation still holds
        assert result.corner_counts[0] * 2 + result.single_counts[0] == 6


class TestPartitionStubsValidation:
    """Tests for input validation and error handling."""

    def test_mixed_cardinality_motif_raises(self, rng: np.random.Generator) -> None:
        """Motifs with non-uniform cardinality are rejected."""
        stubs = np.array([6, 8, 4, 10])

        with pytest.raises(MixedCardinalityError):
            partition_stubs(stubs, G7, phi=0.5, rng=rng)

    def test_phi_negative_raises(self, rng: np.random.Generator) -> None:
        """Negative phi is rejected."""
        stubs = np.array([6, 8, 4])

        with pytest.raises(ValueError, match="phi"):
            partition_stubs(stubs, G2, phi=-0.1, rng=rng)

    def test_phi_greater_than_one_raises(self, rng: np.random.Generator) -> None:
        """phi > 1 is rejected."""
        stubs = np.array([6, 8, 4])

        with pytest.raises(ValueError, match="phi"):
            partition_stubs(stubs, G2, phi=1.1, rng=rng)


class TestStubPartitionProperties:
    """Tests for StubPartition computed properties."""

    def test_total_corners_matches_sum(self, rng: np.random.Generator) -> None:
        """total_corners property equals corner_counts.sum()."""
        stubs = np.array([6, 8, 4, 10])

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert result.total_corners == result.corner_counts.sum()

    def test_total_singles_matches_sum(self, rng: np.random.Generator) -> None:
        """total_singles property equals single_counts.sum()."""
        stubs = np.array([6, 8, 4, 10])

        result = partition_stubs(stubs, G2, phi=0.5, rng=rng)

        assert result.total_singles == result.single_counts.sum()


class TestPartitionStubsReproducibility:
    """Tests for deterministic behavior with seeded generators."""

    def test_same_seed_same_result(self) -> None:
        """Identical seeds produce identical partitions."""
        stubs = np.array([6, 8, 4, 10, 12])

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        result1 = partition_stubs(stubs, G2, phi=0.5, rng=rng1)
        result2 = partition_stubs(stubs, G2, phi=0.5, rng=rng2)

        np.testing.assert_array_equal(result1.corner_counts, result2.corner_counts)
        np.testing.assert_array_equal(result1.single_counts, result2.single_counts)
        np.testing.assert_array_equal(result1.corners, result2.corners)
        np.testing.assert_array_equal(result1.singles, result2.singles)

    def test_different_seeds_different_results(self) -> None:
        """Different seeds produce different partitions (with high probability)."""
        stubs = np.array([10, 10, 10, 10, 10])

        rng1 = np.random.default_rng(111)
        rng2 = np.random.default_rng(222)

        result1 = partition_stubs(stubs, G2, phi=0.5, rng=rng1)
        result2 = partition_stubs(stubs, G2, phi=0.5, rng=rng2)

        # At least one count should differ
        assert not np.array_equal(result1.corner_counts, result2.corner_counts)
