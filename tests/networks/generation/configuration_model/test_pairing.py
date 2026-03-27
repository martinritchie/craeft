"""Tests for pairing module."""

import numpy as np
import pytest

from craeft.networks.generation.configuration_model.pairing import (
    Erased,
    PairingResult,
    Refuse,
    Repeated,
    _canonical_edge,
    _is_valid_pair,
    get_pairer,
    pair_singles,
)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


# === Parametrized across all strategies ===
@pytest.mark.parametrize("strategy", ["repeated", "refuse", "erased"])
class TestAllStrategies:
    def test_empty_singles(self, strategy: str, rng: np.random.Generator) -> None:
        """Empty/single stub returns zero edges."""
        for singles in [np.array([], dtype=np.int_), np.array([0])]:
            result = pair_singles(singles, set(), rng, strategy)  # type: ignore[arg-type]
            assert result.num_edges == 0
            assert result.unpaired == len(singles)

    def test_basic_pairing(self, strategy: str, rng: np.random.Generator) -> None:
        """Simple case: 4 distinct nodes pair to 2 edges."""
        singles = np.array([0, 1, 2, 3])
        result = pair_singles(singles, set(), rng, strategy)  # type: ignore[arg-type]
        assert result.num_edges == 2
        assert result.unpaired == 0

    def test_avoids_existing_edges(
        self, strategy: str, rng: np.random.Generator
    ) -> None:
        """No edge duplicates existing motif edges."""
        singles = np.array([0, 1, 0, 1, 2, 3])  # Force some (0,1) attempts
        existing = {(0, 1)}
        result = pair_singles(singles, existing, rng, strategy)  # type: ignore[arg-type]
        edges = {
            _canonical_edge(int(r), int(c)) for r, c in zip(result.rows, result.cols)
        }
        assert (0, 1) not in edges

    def test_odd_count_drops_one(self, strategy: str, rng: np.random.Generator) -> None:
        """Odd stub count leaves at least 1 unpaired."""
        singles = np.array([0, 1, 2, 3, 4])
        result = pair_singles(singles, set(), rng, strategy)  # type: ignore[arg-type]
        assert result.unpaired >= 1


# === Strategy-specific edge cases ===
class TestRepeated:
    def test_retries_on_collision(self, rng: np.random.Generator) -> None:
        """Swaps to avoid self-loop when alternatives exist."""
        singles = np.array([0, 0, 1, 2])
        result = Repeated().pair(singles, set(), rng)
        assert all(r != c for r, c in zip(result.rows, result.cols))

    def test_exhausted_attempts_all_selfloops(self, rng: np.random.Generator) -> None:
        """Returns empty when all pairs are self-loops."""
        singles = np.array([0, 0, 0, 0])
        result = Repeated(max_attempts=10).pair(singles, set(), rng)
        assert result.num_edges == 0


class TestRefuse:
    def test_exhausted_batch_rejections(self, rng: np.random.Generator) -> None:
        """Returns empty when no valid batch found."""
        singles = np.array([0, 0])
        result = Refuse(max_attempts=10).pair(singles, set(), rng)
        assert result.num_edges == 0
        assert result.unpaired == 2


class TestErased:
    def test_filters_self_loops_and_existing(self, rng: np.random.Generator) -> None:
        """Filters out self-loops and existing edges."""
        singles = np.array([0, 0, 0, 1])  # Mostly self-loops
        existing = {(0, 1)}
        result = Erased().pair(singles, existing, rng)
        edges = {
            _canonical_edge(int(r), int(c)) for r, c in zip(result.rows, result.cols)
        }
        assert (0, 1) not in edges
        assert all(r != c for r, c in zip(result.rows, result.cols))


# === Helpers & Factory ===
def test_canonical_edge() -> None:
    assert _canonical_edge(5, 3) == (3, 5)
    assert _canonical_edge(2, 7) == (2, 7)


def test_is_valid_pair() -> None:
    assert not _is_valid_pair(1, 1, set(), set())  # self-loop
    assert not _is_valid_pair(1, 2, {(1, 2)}, set())  # existing
    assert not _is_valid_pair(2, 3, set(), {(2, 3)})  # formed
    assert _is_valid_pair(1, 2, set(), set())  # valid


def test_get_pairer_valid() -> None:
    assert isinstance(get_pairer("repeated"), Repeated)
    assert isinstance(get_pairer("refuse"), Refuse)
    assert isinstance(get_pairer("erased"), Erased)


def test_get_pairer_invalid() -> None:
    with pytest.raises(ValueError, match="Unknown pairer"):
        get_pairer("invalid")  # type: ignore[arg-type]


def test_pairing_result_num_edges() -> None:
    result = PairingResult(np.array([0, 1]), np.array([2, 3]), unpaired=0)
    assert result.num_edges == 2
