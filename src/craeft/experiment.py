"""Experiment orchestration for network-epidemic pipelines.

Composes a NetworkGenerator with an EpidemicSimulator, running repeated
trials across independently generated networks. Supports parallel
execution via multiprocessing.
"""

from dataclasses import dataclass

import numpy as np

from craeft.networks.generation.generator import NetworkGenerator
from craeft.point_processes.epidemics.simulator import EpidemicSimulator
from craeft.point_processes.gillespie import EnsembleResult


@dataclass(frozen=True)
class ExperimentResult:
    """Collection of per-network simulation results."""

    results: tuple[EnsembleResult, ...]

    def __len__(self) -> int:
        return len(self.results)

    def __iter__(self):
        return iter(self.results)

    def __getitem__(self, index: int) -> EnsembleResult:
        return self.results[index]


def _run_trial(
    args: tuple[NetworkGenerator, EpidemicSimulator, int, int],
) -> EnsembleResult:
    """Worker function: generate one network and run epidemic ensemble.

    Defined at module level so it is picklable for multiprocessing.
    """
    generator, simulator, gen_seed, sim_seed = args
    adjacency = generator.generate(np.random.default_rng(gen_seed))
    return simulator.run(adjacency, np.random.default_rng(sim_seed))


@dataclass(frozen=True)
class Experiment:
    """Compose a generator and simulator into a repeatable experiment.

    Each call to ``run()`` generates ``n_networks`` independent networks
    and runs a converged epidemic ensemble on each. Trials are independent
    and can be parallelised across workers.

    Example:
        >>> generator = ErdosRenyiGenerator(n=500, p=0.01)
        >>> simulator = SIRSimulator(config=SIRConfig(tau=0.5, gamma=1.0))
        >>> experiment = Experiment(generator, simulator, n_networks=50)
        >>> result = experiment.run(np.random.default_rng(42), n_workers=4)
        >>> len(result)
        50
    """

    generator: NetworkGenerator
    simulator: EpidemicSimulator
    n_networks: int

    def run(
        self,
        rng: np.random.Generator,
        n_workers: int = 1,
    ) -> ExperimentResult:
        """Execute the experiment.

        Args:
            rng: Root random generator (used to derive per-trial seeds).
            n_workers: Number of parallel workers. 1 = sequential.

        Returns:
            One EnsembleResult per network, wrapped in ExperimentResult.
        """
        args = [
            (
                self.generator,
                self.simulator,
                int(rng.integers(2**31)),
                int(rng.integers(2**31)),
            )
            for _ in range(self.n_networks)
        ]

        if n_workers <= 1:
            results = [_run_trial(a) for a in args]
        else:
            results = self._run_parallel(args, n_workers)

        return ExperimentResult(results=tuple(results))

    @staticmethod
    def _run_parallel(
        args: list[tuple[NetworkGenerator, EpidemicSimulator, int, int]],
        n_workers: int,
    ) -> list[EnsembleResult]:
        """Dispatch trials across a process pool."""
        from concurrent.futures import ProcessPoolExecutor

        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            return list(executor.map(_run_trial, args))
