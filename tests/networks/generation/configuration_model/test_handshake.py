"""Tests for handshake lemma balancing."""

import numpy as np

from craeft.networks.generation.configuration_model.handshake import (
    balance_allocation,
)
from craeft.networks.generation.configuration_model.hyperstub_matrix import (
    HyperstubAllocation,
)
from craeft.networks.generation.configuration_model.subgraph_spec import SubgraphSpec
from craeft.networks.generation.motifs import G2


class TestBalanceAllocation:
    """Tests for balance_allocation."""

    def test_already_balanced_unchanged(self) -> None:
        """Balanced allocation passes through."""
        spec = SubgraphSpec(motif=G2, sequence=np.array([3, 3, 3]))
        bins = {(0, 1): np.array([3, 3, 3])}  # total=9, group_size=3, 9%3=0
        alloc = HyperstubAllocation(
            bins=bins,
            singles=np.array([0, 0, 0]),
            specs=(spec,),
        )

        balanced = balance_allocation(alloc)
        np.testing.assert_array_equal(balanced.bins[(0, 1)], [3, 3, 3])

    def test_sheds_excess_to_singles(self) -> None:
        """Excess corners are returned as singles."""
        spec = SubgraphSpec(motif=G2, sequence=np.array([4, 4, 4]))
        # total=10, group_size=3 (G2 has 3 corners of type 1), 10%3=1
        bins = {(0, 1): np.array([4, 3, 3])}
        alloc = HyperstubAllocation(
            bins=bins,
            singles=np.array([0, 2, 2]),
            specs=(spec,),
        )

        balanced = balance_allocation(alloc)
        total = int(balanced.bins[(0, 1)].sum())
        assert total % 3 == 0

    def test_even_singles(self) -> None:
        """Singles total is made even."""
        spec = SubgraphSpec(motif=G2, sequence=np.array([3, 3, 3]))
        bins = {(0, 1): np.array([3, 3, 3])}
        alloc = HyperstubAllocation(
            bins=bins,
            singles=np.array([3, 2, 2]),  # total=7, odd
            specs=(spec,),
        )

        balanced = balance_allocation(alloc)
        assert int(balanced.singles.sum()) % 2 == 0
