"""Tests for orbit-aware connectors."""

import numpy as np

from craeft.networks.generation.configuration_model.orbit_connector import (
    OrbitErased,
    OrbitRefuse,
    OrbitRepeated,
    get_orbit_connector,
)
from craeft.networks.generation.motifs import G2, G7


class TestOrbitConnectorComplete:
    """For complete motifs, orbit connectors should still work."""

    def test_triangle_produces_edges(self, rng: np.random.Generator) -> None:
        """G2 with single bin type produces triangle edges."""
        # 6 nodes, each appears once -> 2 triangles
        bins = {1: np.array([0, 1, 2, 3, 4, 5])}
        connector = OrbitRepeated()
        rows, cols = connector.connect_orbit(bins, G2, rng)

        assert len(rows) > 0
        assert len(rows) == len(cols)


class TestOrbitConnectorIncomplete:
    """For incomplete motifs with typed bins."""

    def test_g7_diamond_uses_both_types(self, rng: np.random.Generator) -> None:
        """G7 has type_counts {1: 2, 2: 2}, requiring two bins."""
        # 4 nodes per instance, 2 instances -> 8 slots across two types
        bins = {
            1: np.array([0, 1, 2, 3]),  # type 1 (degree 2): 2 per instance
            2: np.array([4, 5, 6, 7]),  # type 2 (degree 3): 2 per instance
        }
        connector = OrbitRepeated()
        rows, cols = connector.connect_orbit(bins, G7, rng)

        assert len(rows) > 0

    def test_empty_bins_produce_no_edges(self, rng: np.random.Generator) -> None:
        bins = {1: np.array([], dtype=np.int_), 2: np.array([], dtype=np.int_)}
        connector = OrbitRepeated()
        rows, cols = connector.connect_orbit(bins, G7, rng)

        assert len(rows) == 0

    def test_refuse_returns_empty_on_collision(self, rng: np.random.Generator) -> None:
        """When all nodes collide, refuse returns empty."""
        # Same node in both bins -> guaranteed collision
        bins = {1: np.array([0, 0]), 2: np.array([0, 0])}
        connector = OrbitRefuse(max_attempts=10)
        rows, cols = connector.connect_orbit(bins, G7, rng)

        assert len(rows) == 0

    def test_erased_removes_self_loops(self, rng: np.random.Generator) -> None:
        """Erased cleans up collisions post-hoc."""
        bins = {
            1: np.array([0, 1]),
            2: np.array([2, 3]),
        }
        connector = OrbitErased()
        rows, cols = connector.connect_orbit(bins, G7, rng)

        # No self-loops
        for r, c in zip(rows, cols):
            assert r != c


class TestGetOrbitConnector:
    """Tests for the factory function."""

    def test_repeated(self) -> None:
        assert isinstance(get_orbit_connector("repeated"), OrbitRepeated)

    def test_refuse(self) -> None:
        assert isinstance(get_orbit_connector("refuse"), OrbitRefuse)

    def test_erased(self) -> None:
        assert isinstance(get_orbit_connector("erased"), OrbitErased)
