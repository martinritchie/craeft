"""Tests for multinomial decomposition."""

import numpy as np

from craeft.networks.generation.configuration_model.multinomial_partition import (
    compute_stub_cost,
    multinomial_decompose,
)
from craeft.networks.generation.configuration_model.subgraph_spec import SubgraphSpec
from craeft.networks.generation.motifs import G2, G7, G8, G14


class TestMultinomialDecomposeComplete:
    """Tests for complete (uniform-cardinality) motifs."""

    def test_complete_motif_single_corner_type(self, rng: np.random.Generator) -> None:
        spec = SubgraphSpec(motif=G2, sequence=np.array([3, 2, 1]))
        result = multinomial_decompose(spec, rng)

        assert len(result) == 1
        corner_type = G2.corner_types[0]
        np.testing.assert_array_equal(result[corner_type], [3, 2, 1])

    def test_k4_complete_single_type(self, rng: np.random.Generator) -> None:
        spec = SubgraphSpec(motif=G8, sequence=np.full(10, 4))
        result = multinomial_decompose(spec, rng)

        assert len(result) == 1
        assert result[G8.corner_types[0]].sum() == 40


class TestMultinomialDecomposeIncomplete:
    """Tests for incomplete (mixed-cardinality) motifs."""

    def test_g7_diamond_two_corner_types(self, rng: np.random.Generator) -> None:
        """G7 has types {1: degree 2, 2: degree 3} with counts {1: 2, 2: 2}."""
        spec = SubgraphSpec(motif=G7, sequence=np.full(100, 4))
        result = multinomial_decompose(spec, rng)

        assert len(result) == 2
        assert 1 in result
        assert 2 in result

    def test_stub_conservation(self, rng: np.random.Generator) -> None:
        """Total across types per node equals original sequence."""
        spec = SubgraphSpec(motif=G7, sequence=np.full(50, 6))
        result = multinomial_decompose(spec, rng)

        total = sum(counts for counts in result.values())
        np.testing.assert_array_equal(total, spec.sequence)

    def test_handshake_divisibility(self, rng: np.random.Generator) -> None:
        """Each type total must be divisible by its group size."""
        spec = SubgraphSpec(motif=G14, sequence=np.full(100, 3))
        result = multinomial_decompose(spec, rng)

        for corner_type, counts in result.items():
            group_size = G14.type_counts[corner_type]
            assert int(counts.sum()) % group_size == 0

    def test_zero_sequence_produces_zeros(self, rng: np.random.Generator) -> None:
        spec = SubgraphSpec(motif=G7, sequence=np.zeros(20, dtype=np.int_))
        result = multinomial_decompose(spec, rng)

        for counts in result.values():
            assert counts.sum() == 0


class TestComputeStubCost:
    """Tests for stub cost calculation."""

    def test_triangle_cost_is_two_per_corner(self) -> None:
        """G2 cardinality is 2, so cost = 2 * count."""
        decomp = {1: np.array([3, 2, 1])}
        cost = compute_stub_cost(decomp, G2)
        np.testing.assert_array_equal(cost, [6, 4, 2])

    def test_mixed_cardinality_cost(self) -> None:
        """G7 has cardinalities {1: 2, 2: 3}."""
        decomp = {1: np.array([2, 1]), 2: np.array([1, 2])}
        cost = compute_stub_cost(decomp, G7)
        # node 0: 2*2 + 1*3 = 7, node 1: 1*2 + 2*3 = 8
        np.testing.assert_array_equal(cost, [7, 8])
