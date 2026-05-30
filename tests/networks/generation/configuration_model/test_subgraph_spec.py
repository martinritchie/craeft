"""Tests for SubgraphSpec."""

import numpy as np
import pytest

from craeft.networks.generation.configuration_model.subgraph_spec import SubgraphSpec
from craeft.networks.generation.motifs import G2, G8


class TestSubgraphSpec:
    """Tests for SubgraphSpec validation and properties."""

    def test_total_sums_sequence(self) -> None:
        spec = SubgraphSpec(motif=G2, sequence=np.array([2, 3, 1]))
        assert spec.total == 6

    def test_n_nodes_matches_sequence_length(self) -> None:
        spec = SubgraphSpec(motif=G2, sequence=np.full(100, 2))
        assert spec.n_nodes == 100

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            SubgraphSpec(motif=G2, sequence=np.array([1, -1, 2]))

    def test_zero_sequence_is_valid(self) -> None:
        spec = SubgraphSpec(motif=G8, sequence=np.zeros(50, dtype=np.int_))
        assert spec.total == 0

    def test_frozen(self) -> None:
        spec = SubgraphSpec(motif=G2, sequence=np.array([1, 2]))
        with pytest.raises(AttributeError):
            spec.motif = G8  # type: ignore[misc]
