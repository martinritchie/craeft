"""Tests for HyperstubAllocation and build_allocation."""

import numpy as np
import pytest

from craeft.networks.generation.configuration_model.hyperstub_matrix import (
    HyperstubAllocation,
    build_allocation,
)
from craeft.networks.generation.configuration_model.subgraph_spec import SubgraphSpec
from craeft.networks.generation.motifs import G2


class TestBuildAllocation:
    """Tests for build_allocation."""

    def test_stub_conservation(self) -> None:
        """Total cost + singles = original degrees."""
        degrees = np.array([6, 6, 6, 6])
        spec = SubgraphSpec(motif=G2, sequence=np.array([1, 1, 1, 1]))
        decomp = {1: np.array([1, 1, 1, 1])}

        alloc = build_allocation(degrees, [spec], [decomp])

        # G2 cardinality is 2, so cost per node = 1 * 2 = 2
        expected_singles = degrees - 2
        np.testing.assert_array_equal(alloc.singles, expected_singles)

    def test_rejects_overcost(self) -> None:
        """Raises if stub cost exceeds degree."""
        degrees = np.array([2, 2])
        spec = SubgraphSpec(motif=G2, sequence=np.array([3, 3]))
        decomp = {1: np.array([3, 3])}  # cost = 6 > degree 2

        with pytest.raises(ValueError, match="exceeds degree"):
            build_allocation(degrees, [spec], [decomp])


class TestHyperstubAllocation:
    """Tests for HyperstubAllocation methods."""

    def test_node_ids_for_repeats_correctly(self) -> None:
        bins = {(0, 1): np.array([3, 0, 2])}
        alloc = HyperstubAllocation(
            bins=bins,
            singles=np.array([1, 1, 1]),
            specs=(),
        )

        ids = alloc.node_ids_for(0, 1)
        # node 0 appears 3 times, node 1 appears 0, node 2 appears 2
        assert list(ids) == [0, 0, 0, 2, 2]

    def test_single_stubs_builds_flat_array(self) -> None:
        alloc = HyperstubAllocation(
            bins={},
            singles=np.array([2, 0, 3]),
            specs=(),
        )

        stubs = alloc.single_stubs()
        assert list(stubs) == [0, 0, 2, 2, 2]
