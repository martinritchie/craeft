"""Generic Gillespie algorithm for continuous-time point processes.

Exact stochastic simulation engine that works with any process implementing
the ``ContinuousTimeProcess`` interface. Provides convergence-based ensemble
execution, trajectory aggregation, and parallel dispatch.
"""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from craeft.point_processes.process import (
    ProcessFactory,
    Trajectory,
)

# Type for progress callback: (n_completed, max_n, relative_sem, converged) -> None
ProgressCallback = Callable[[int, int, float, bool], None]


@dataclass(frozen=True)
class ConvergenceInfo:
    """Metadata about convergence of a stochastic ensemble."""

    converged: bool
    n_realizations: int
    relative_sem: float
    threshold: float
    n_discarded: int = 0


@dataclass(frozen=True)
class EnsembleResult:
    """Aggregated statistics from a stochastic simulation ensemble.

    Compartment time series are stored in dicts keyed by compartment name,
    making this type process-agnostic.
    """

    t: NDArray[np.float64]
    means: dict[str, NDArray[np.float64]]
    stds: dict[str, NDArray[np.float64]]
    scalar_outputs: NDArray[np.float64]
    convergence: ConvergenceInfo

    @property
    def scalar_output_mean(self) -> float:
        """Mean of the scalar convergence quantity."""
        return float(np.mean(self.scalar_outputs))

    @property
    def scalar_output_std(self) -> float:
        """Standard deviation of the scalar convergence quantity."""
        return float(np.std(self.scalar_outputs))


class ConvergenceMonitor:
    """Track running statistics for convergence detection using Welford's algorithm.

    Welford's algorithm computes running mean and variance in a single pass
    with numerical stability, avoiding catastrophic cancellation.
    """

    _n: int
    _mean: float
    _m2: float
    _threshold: float

    def __init__(self, threshold: float) -> None:
        self._n = 0
        self._mean = 0.0
        self._m2 = 0.0
        self._threshold = threshold

    def update(self, value: float) -> None:
        """Add a new observation using Welford's online algorithm."""
        self._n += 1
        delta = value - self._mean
        self._mean += delta / self._n
        delta2 = value - self._mean
        self._m2 += delta * delta2

    @property
    def n(self) -> int:
        """Number of observations."""
        return self._n

    @property
    def mean(self) -> float:
        """Current running mean."""
        return self._mean

    @property
    def variance(self) -> float:
        """Current sample variance (n-1 denominator)."""
        if self._n < 2:
            return float("inf")
        return self._m2 / (self._n - 1)

    @property
    def std(self) -> float:
        """Current sample standard deviation."""
        return float(np.sqrt(self.variance))

    @property
    def sem(self) -> float:
        """Standard error of the mean."""
        if self._n < 2:
            return float("inf")
        return float(self.std / np.sqrt(self._n))

    @property
    def relative_sem(self) -> float:
        """Relative standard error (SEM / mean)."""
        if self._n < 2 or self._mean == 0:
            return float("inf")
        return self.sem / abs(self._mean)

    def is_converged(self, min_n: int) -> bool:
        """Check if convergence criterion is met."""
        if self._n < min_n:
            return False
        return self.relative_sem < self._threshold


def run_once(
    factory: ProcessFactory,
    t_end: float,
    rng: np.random.Generator,
) -> tuple[Trajectory, float, bool]:
    """Run a single realisation of the Gillespie algorithm.

    Args:
        factory: Creates a fresh process instance for this realisation.
        t_end: Maximum simulation time.
        rng: Random number generator for event sampling.

    Returns:
        (trajectory, scalar_output, should_accept) tuple.
    """
    process = factory.create(rng)

    while not process.is_absorbing() and process.time < t_end:
        all_rates = process.rates()
        total_rate = all_rates.sum()
        dt = rng.exponential(1.0 / total_rate)

        if process.time + dt > t_end:
            break

        event_idx = int(rng.choice(len(all_rates), p=all_rates / total_rate))
        process.execute(event_idx, dt)

    return process.trajectory(), process.scalar_output(), process.should_accept()


def _run_single_worker(
    args: tuple[ProcessFactory, float, int],
) -> tuple[Trajectory, float, bool]:
    """Worker function for multiprocessing (must be picklable at module level)."""
    factory, t_end, seed = args
    rng = np.random.default_rng(seed)
    return run_once(factory, t_end, rng)


def simulate(
    factory: ProcessFactory,
    rng: np.random.Generator | None = None,
    n_points: int = 200,
    progress_callback: ProgressCallback | None = None,
) -> EnsembleResult:
    """Run stochastic simulation ensemble with convergence-based stopping.

    Uses the Gillespie algorithm for exact stochastic simulation. Runs
    until the relative standard error of the scalar output drops below
    the configured threshold.

    Args:
        factory: Creates fresh process instances for each realisation.
        rng: Random number generator (used for seed generation in parallel mode).
        n_points: Number of time points for output grid.
        progress_callback: Optional callback for progress updates.
            Called with (n_completed, max_n, relative_sem, converged).

    Returns:
        EnsembleResult with ensemble statistics and convergence info.
    """
    rng = rng or np.random.default_rng()
    config = factory.convergence_config

    if config.num_workers > 1:
        return _simulate_parallel(factory, rng, n_points, progress_callback)
    return _simulate_sequential(factory, rng, n_points, progress_callback)


def _simulate_sequential(
    factory: ProcessFactory,
    rng: np.random.Generator,
    n_points: int,
    progress_callback: ProgressCallback | None = None,
) -> EnsembleResult:
    """Run simulations sequentially (single worker)."""
    config = factory.convergence_config
    monitor = ConvergenceMonitor(config.convergence_threshold)
    trajectories: list[Trajectory] = []
    scalars: list[float] = []
    n_total = 0
    n_discarded = 0

    while not monitor.is_converged(config.min_realizations):
        if n_total >= config.max_realizations:
            break

        trajectory, scalar, accepted = run_once(factory, config.t_end, rng)
        n_total += 1

        if accepted:
            trajectories.append(trajectory)
            scalars.append(scalar)
            monitor.update(scalar)
        else:
            n_discarded += 1

        if progress_callback:
            progress_callback(
                n_total,
                config.max_realizations,
                monitor.relative_sem,
                monitor.is_converged(config.min_realizations),
            )

    convergence = ConvergenceInfo(
        converged=monitor.is_converged(config.min_realizations),
        n_realizations=monitor.n,
        relative_sem=monitor.relative_sem,
        threshold=config.convergence_threshold,
        n_discarded=n_discarded,
    )

    if not trajectories:
        return _empty_result(trajectories, config.t_end, n_points, convergence)

    return _aggregate_trajectories(
        trajectories, scalars, config.t_end, n_points, convergence
    )


def _simulate_parallel(
    factory: ProcessFactory,
    rng: np.random.Generator,
    n_points: int,
    progress_callback: ProgressCallback | None = None,
) -> EnsembleResult:
    """Run simulations in parallel using multiprocessing."""
    from concurrent.futures import ProcessPoolExecutor

    config = factory.convergence_config
    monitor = ConvergenceMonitor(config.convergence_threshold)
    trajectories: list[Trajectory] = []
    scalars: list[float] = []
    batch_size = config.num_workers
    n_total = 0
    n_discarded = 0

    with ProcessPoolExecutor(max_workers=config.num_workers) as executor:
        while not monitor.is_converged(config.min_realizations):
            remaining = config.max_realizations - n_total
            if remaining <= 0:
                break

            current_batch = min(batch_size, remaining)
            seeds = rng.integers(0, 2**31, size=current_batch).tolist()
            args = [(factory, config.t_end, seed) for seed in seeds]

            batch_results = list(executor.map(_run_single_worker, args))

            for trajectory, scalar, accepted in batch_results:
                n_total += 1

                if accepted:
                    trajectories.append(trajectory)
                    scalars.append(scalar)
                    monitor.update(scalar)
                else:
                    n_discarded += 1

                if progress_callback:
                    progress_callback(
                        n_total,
                        config.max_realizations,
                        monitor.relative_sem,
                        monitor.is_converged(config.min_realizations),
                    )

    convergence = ConvergenceInfo(
        converged=monitor.is_converged(config.min_realizations),
        n_realizations=monitor.n,
        relative_sem=monitor.relative_sem,
        threshold=config.convergence_threshold,
        n_discarded=n_discarded,
    )

    if not trajectories:
        return _empty_result(trajectories, config.t_end, n_points, convergence)

    return _aggregate_trajectories(
        trajectories, scalars, config.t_end, n_points, convergence
    )


def _empty_result(
    trajectories: list[Trajectory],
    t_end: float,
    n_points: int,
    convergence: ConvergenceInfo,
) -> EnsembleResult:
    """Return a valid zero-valued EnsembleResult when all runs were filtered."""
    t_grid = np.linspace(0, t_end, n_points)
    zeros = np.zeros(n_points)

    # Derive compartment names from the factory's process if possible,
    # otherwise fall back to empty dicts
    if trajectories:
        names = trajectories[0].names
    else:
        names = []

    return EnsembleResult(
        t=t_grid,
        means={name: zeros.copy() for name in names},
        stds={name: zeros.copy() for name in names},
        scalar_outputs=np.array([], dtype=np.float64),
        convergence=convergence,
    )


def _interpolate_trajectory(
    trajectory: Trajectory,
    t_grid: NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """Interpolate a trajectory to a common time grid.

    Uses step interpolation (piecewise constant) since compartment
    counts change discretely at event times.

    Args:
        trajectory: Single simulation trajectory.
        t_grid: Common time points to interpolate to.

    Returns:
        Dict mapping compartment names to interpolated arrays.
    """
    indices = np.searchsorted(trajectory.t, t_grid, side="right") - 1
    indices = np.clip(indices, 0, len(trajectory.t) - 1)

    return {
        name: values[indices].astype(np.float64)
        for name, values in trajectory.compartments.items()
    }


def _aggregate_trajectories(
    trajectories: list[Trajectory],
    scalar_outputs: list[float],
    t_end: float,
    n_points: int,
    convergence: ConvergenceInfo,
) -> EnsembleResult:
    """Aggregate ensemble of trajectories into summary statistics.

    Args:
        trajectories: List of simulation trajectories.
        scalar_outputs: Scalar output from each realisation.
        t_end: End time for the time grid.
        n_points: Number of points in the time grid.
        convergence: Convergence metadata from the simulation run.

    Returns:
        EnsembleResult with mean and std per compartment over time.
    """
    t_grid = np.linspace(0, t_end, n_points)
    n_runs = len(trajectories)
    names = trajectories[0].names

    # Preallocate arrays for interpolated values
    all_data: dict[str, NDArray[np.float64]] = {
        name: np.zeros((n_runs, n_points)) for name in names
    }

    for idx, traj in enumerate(trajectories):
        interpolated = _interpolate_trajectory(traj, t_grid)
        for name in names:
            all_data[name][idx] = interpolated[name]

    means = {name: data.mean(axis=0) for name, data in all_data.items()}
    stds = {name: data.std(axis=0) for name, data in all_data.items()}

    return EnsembleResult(
        t=t_grid,
        means=means,
        stds=stds,
        scalar_outputs=np.array(scalar_outputs, dtype=np.float64),
        convergence=convergence,
    )
