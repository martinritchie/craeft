"""Tests for a priori clustering prediction."""

import numpy as np

from craeft.networks.generation.configuration_model.clustering_prediction import (
    predict_clustering,
)
from craeft.networks.generation.configuration_model.subgraph_spec import SubgraphSpec
from craeft.networks.generation.motifs import G2, G5, G8


class TestPredictClustering:
    """Tests for predict_clustering."""

    def test_zero_specs_returns_zero(self) -> None:
        degrees = np.full(100, 5)
        assert predict_clustering(degrees, []) == 0.0

    def test_triangles_positive(self) -> None:
        """G2 (triangle) has 1 triangle per instance."""
        degrees = np.full(100, 5)
        spec = SubgraphSpec(motif=G2, sequence=np.full(100, 2))

        cc = predict_clustering(degrees, [spec])
        assert cc > 0.0

    def test_k4_more_than_triangle(self) -> None:
        """G8 (K4) has 4 triangles vs G2's 1, so same count gives higher clustering."""
        degrees = np.full(100, 6)
        spec_g2 = SubgraphSpec(motif=G2, sequence=np.full(100, 1))
        spec_g8 = SubgraphSpec(motif=G8, sequence=np.full(100, 1))

        cc_g2 = predict_clustering(degrees, [spec_g2])
        cc_g8 = predict_clustering(degrees, [spec_g8])

        assert cc_g8 > cc_g2

    def test_cycle_no_triangles(self) -> None:
        """G5 (4-cycle) contains no triangles, so contributes zero clustering."""
        degrees = np.full(100, 4)
        spec = SubgraphSpec(motif=G5, sequence=np.full(100, 2))

        cc = predict_clustering(degrees, [spec])
        assert cc == 0.0

    def test_capped_at_one(self) -> None:
        """Prediction should never exceed 1.0."""
        degrees = np.full(10, 5)
        spec = SubgraphSpec(motif=G8, sequence=np.full(10, 100))

        cc = predict_clustering(degrees, [spec])
        assert cc <= 1.0

    def test_zero_degree_returns_zero(self) -> None:
        degrees = np.zeros(10, dtype=np.int_)
        spec = SubgraphSpec(motif=G2, sequence=np.zeros(10, dtype=np.int_))

        assert predict_clustering(degrees, [spec]) == 0.0
