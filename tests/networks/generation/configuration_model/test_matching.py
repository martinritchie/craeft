"""Tests for greedy cardinality matching."""

import numpy as np

from craeft.networks.generation.configuration_model.matching import cardinality_match
from craeft.networks.generation.configuration_model.multinomial_partition import (
    multinomial_decompose,
)
from craeft.networks.generation.configuration_model.subgraph_spec import SubgraphSpec
from craeft.networks.generation.motifs import G2, G8


class TestCardinalityMatch:
    """Tests for the greedy matching algorithm."""

    def test_no_node_over_allocated(self, rng: np.random.Generator) -> None:
        """Total allocated stubs never exceeds original degree."""
        degrees = np.full(100, 6)
        spec = SubgraphSpec(motif=G2, sequence=np.full(100, 2))
        decomp = multinomial_decompose(spec, rng)

        alloc = cardinality_match(degrees, [spec], [decomp], rng)

        total_cost = np.zeros_like(degrees)
        for (_, ct), counts in alloc.bins.items():
            total_cost += counts * spec.motif.cardinalities[ct]
        total_used = total_cost + alloc.singles

        assert np.all(total_used <= degrees)

    def test_heterogeneous_degrees(self, rng: np.random.Generator) -> None:
        """Works with mixed degree sequence."""
        degrees = np.array([10, 4, 8, 3, 6, 5, 7, 2, 9, 4])
        spec = SubgraphSpec(motif=G2, sequence=np.full(10, 1))
        decomp = multinomial_decompose(spec, rng)

        alloc = cardinality_match(degrees, [spec], [decomp], rng)

        # Singles should be non-negative
        assert np.all(alloc.singles >= 0)

    def test_empty_spec_preserves_degrees(self, rng: np.random.Generator) -> None:
        """Zero sequences leave degrees untouched."""
        degrees = np.full(20, 5)
        spec = SubgraphSpec(motif=G2, sequence=np.zeros(20, dtype=np.int_))
        decomp = multinomial_decompose(spec, rng)

        alloc = cardinality_match(degrees, [spec], [decomp], rng)

        np.testing.assert_array_equal(alloc.singles, degrees)

    def test_multiple_specs(self, rng: np.random.Generator) -> None:
        """Multiple subgraph families can coexist."""
        degrees = np.full(100, 10)
        spec_g2 = SubgraphSpec(motif=G2, sequence=np.full(100, 1))
        spec_g8 = SubgraphSpec(motif=G8, sequence=np.full(100, 1))
        decomps = [
            multinomial_decompose(spec_g2, rng),
            multinomial_decompose(spec_g8, rng),
        ]

        alloc = cardinality_match(degrees, [spec_g2, spec_g8], decomps, rng)

        assert np.all(alloc.singles >= 0)
