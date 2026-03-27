"""Epidemic simulator protocol and concrete implementations.

Defines a structural typing contract for epidemic simulation and provides
a frozen dataclass implementation wrapping the Gillespie ensemble.
"""

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.sparse import csr_array, csr_matrix

from craeft.point_processes.gillespie import EnsembleResult
from craeft.point_processes.process import ConvergenceConfig


class EpidemicSimulator(Protocol):
    """Structural interface for epidemic simulation.

    Implementations consume a CSR adjacency matrix (the sole contract
    with network generators) and produce ensemble statistics.
    """

    def run(self, adjacency: csr_matrix, rng: np.random.Generator) -> EnsembleResult:
        """Run a converged epidemic ensemble on the given network.

        Args:
            adjacency: Symmetric adjacency matrix in CSR format.
            rng: Random number generator for this ensemble.

        Returns:
            Ensemble statistics from multiple realisations.
        """
        ...


@dataclass(frozen=True)
class SIRSimulator:
    """SIR epidemic simulation via Gillespie algorithm.

    Wraps the convergence-based ensemble in a convenient interface
    matching the ``EpidemicSimulator`` protocol.

    Each call to ``run()`` executes multiple realisations until the
    relative standard error of the mean final size drops below the
    configured threshold.
    """

    config: "SIRConfig"  # noqa: F821 — forward ref resolved at runtime
    convergence: ConvergenceConfig = ConvergenceConfig()
    n_points: int = 200

    def run(self, adjacency: csr_matrix, rng: np.random.Generator) -> EnsembleResult:
        from craeft.point_processes.epidemics.sir import SIRProcessFactory
        from craeft.point_processes.gillespie import simulate

        factory = SIRProcessFactory(
            config=self.config,
            adjacency=csr_array(adjacency),
            _convergence_config=self.convergence,
        )
        return simulate(factory, rng, n_points=self.n_points)
