"""Tests for connector algorithms."""

import numpy as np
import pytest

from craeft.networks.generation.configuration_model.connectors import (
    Erased,
    Refuse,
    Repeated,
    _has_collision,
    get_connector,
)
from craeft.networks.generation.motifs import Motif


@pytest.fixture
def triangle() -> Motif:
    return Motif(adjacency=[[0, 1, 1], [1, 0, 1], [1, 1, 0]])


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# ─── _has_collision ───────────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "nodes,expected",
    [
        (np.array([0, 1, 2]), False),
        (np.array([0, 0, 1]), True),
    ],
)
def test_has_collision(nodes: np.ndarray, expected: bool) -> None:
    assert _has_collision(nodes) == expected


# ─── All connectors: shared behavior ──────────────────────────────────────────
@pytest.mark.parametrize("connector_cls", [Repeated, Refuse, Erased])
class TestConnectorShared:
    def test_too_few_hyperstubs(
        self, connector_cls: type, triangle: Motif, rng: np.random.Generator
    ) -> None:
        rows, _ = connector_cls().connect(np.array([0, 1]), triangle, rng)
        assert len(rows) == 0

    def test_valid_input(
        self, connector_cls: type, triangle: Motif, rng: np.random.Generator
    ) -> None:
        rows, _ = connector_cls().connect(np.array([0, 1, 2, 3, 4, 5]), triangle, rng)
        assert len(rows) == 6


# ─── Connector-specific branches ──────────────────────────────────────────────
def test_repeated_max_attempts(triangle: Motif, rng: np.random.Generator) -> None:
    rows, _ = Repeated(max_attempts=5).connect(np.array([0, 0, 0]), triangle, rng)
    assert len(rows) == 0


def test_refuse_exhausted(triangle: Motif, rng: np.random.Generator) -> None:
    rows, _ = Refuse(max_attempts=5).connect(
        np.array([0, 0, 0, 1, 1, 1]), triangle, rng
    )
    assert len(rows) == 0


def test_erased_cleans_edges(triangle: Motif, rng: np.random.Generator) -> None:
    rows, cols = Erased().connect(np.array([0, 0, 0, 1, 1, 1]), triangle, rng)
    assert len(rows) == 1 and rows[0] != cols[0]


def test_erased_all_self_loops(rng: np.random.Generator) -> None:
    edge = Motif(adjacency=[[0, 1], [1, 0]])
    rows, _ = Erased().connect(np.array([0, 0]), edge, rng)
    assert len(rows) == 0


# ─── get_connector factory ────────────────────────────────────────────────────
@pytest.mark.parametrize(
    "name,cls",
    [("repeated", Repeated), ("refuse", Refuse), ("erased", Erased)],
)
def test_get_connector_valid(name: str, cls: type) -> None:
    assert isinstance(get_connector(name), cls)


def test_get_connector_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown"):
        get_connector("nope")  # type: ignore
