"""Tests for epidemic simulator protocol implementation."""

import pickle

import numpy as np
from scipy.sparse import csr_matrix

from craeft.point_processes.epidemics.simulator import SIRSimulator
from craeft.point_processes.epidemics.sir import SIRConfig
from craeft.point_processes.gillespie import EnsembleResult
from craeft.point_processes.process import ConvergenceConfig


def _small_connected_graph() -> csr_matrix:
    """Create a small connected graph for testing."""
    from craeft.networks.generation import random_graph

    return random_graph(50, 0.1, np.random.default_rng(42))


class TestSIRSimulator:
    """Tests for SIRSimulator."""

    def test_returns_ensemble_result(self) -> None:
        config = SIRConfig(tau=0.5, gamma=1.0)
        convergence = ConvergenceConfig(min_realizations=10, max_realizations=10)
        simulator = SIRSimulator(config=config, convergence=convergence)
        adjacency = _small_connected_graph()
        result = simulator.run(adjacency, np.random.default_rng(42))
        assert isinstance(result, EnsembleResult)

    def test_accepts_csr_matrix(self) -> None:
        """Generators return csr_matrix; simulator must accept it."""
        config = SIRConfig(tau=0.5, gamma=1.0)
        convergence = ConvergenceConfig(min_realizations=10, max_realizations=10)
        simulator = SIRSimulator(config=config, convergence=convergence)
        adjacency = _small_connected_graph()
        assert isinstance(adjacency, csr_matrix)
        result = simulator.run(adjacency, np.random.default_rng(42))
        assert result.convergence.n_realizations > 0

    def test_n_points_propagated(self) -> None:
        config = SIRConfig(tau=0.5, gamma=1.0)
        convergence = ConvergenceConfig(min_realizations=10, max_realizations=10)
        n_points = 50
        simulator = SIRSimulator(
            config=config, convergence=convergence, n_points=n_points
        )
        adjacency = _small_connected_graph()
        result = simulator.run(adjacency, np.random.default_rng(42))
        assert len(result.t) == n_points

    def test_reproducible_with_same_seed(self) -> None:
        config = SIRConfig(tau=0.5, gamma=1.0)
        convergence = ConvergenceConfig(min_realizations=10, max_realizations=10)
        simulator = SIRSimulator(config=config, convergence=convergence)
        adjacency = _small_connected_graph()
        a = simulator.run(adjacency, np.random.default_rng(99))
        b = simulator.run(adjacency, np.random.default_rng(99))
        assert np.allclose(a.means["i"], b.means["i"])

    def test_picklable(self) -> None:
        config = SIRConfig(tau=0.5, gamma=1.0)
        convergence = ConvergenceConfig(min_realizations=10, max_realizations=10)
        simulator = SIRSimulator(config=config, convergence=convergence, n_points=100)
        restored = pickle.loads(pickle.dumps(simulator))
        assert restored.config.tau == 0.5
        assert restored.n_points == 100
