"""Generic continuous-time point process abstractions.

Defines the interface between a stochastic process (which knows its rates
and transitions) and the Gillespie engine (which knows how to sample events).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class Trajectory:
    """Result from a single stochastic simulation run.

    Stores time points and named compartment counts, allowing any
    process to record its own state variables without the engine
    needing to know what they represent.

    Args:
        t: Monotonically increasing event times.
        compartments: Mapping from compartment name to count time series.
            Each array has the same length as ``t``.
    """

    t: NDArray[np.float64]
    compartments: dict[str, NDArray[np.int_]]

    def __getitem__(self, name: str) -> NDArray[np.int_]:
        """Access a compartment by name, e.g. ``trajectory["infected"]``."""
        return self.compartments[name]

    @property
    def names(self) -> list[str]:
        """Compartment names in insertion order."""
        return list(self.compartments.keys())


@dataclass(frozen=True)
class ConvergenceConfig:
    """Configuration for convergence-based ensemble stopping.

    Controls when the Gillespie engine stops launching new realisations.
    Process-agnostic: knows nothing about SIR, Hawkes, etc.
    """

    t_end: float = 15.0
    convergence_threshold: float = 0.02
    min_realizations: int = 30
    max_realizations: int = 10_000
    num_workers: int = 1


class ContinuousTimeProcess(ABC):
    """A continuous-time Markov chain on a network.

    Implementations define the *dynamics* (rates and transitions) while the
    Gillespie engine handles the *mechanics* (exponential waiting times,
    event selection, convergence).

    The process owns its mutable state internally. The engine interacts
    with it only through this interface.

    Lifecycle (managed by the engine):
        1. Engine calls ``rates()`` to get current event rates.
        2. Engine samples dt ~ Exp(1 / total_rate) and selects an event index.
        3. Engine calls ``execute(event_index, dt)`` to apply the transition.
        4. Repeat until ``is_absorbing()`` returns True or t >= t_end.
        5. Engine calls ``trajectory()`` to extract the recorded time series.

    To define a new process, implement all abstract methods and provide
    a factory that constructs fresh instances (one per realisation).
    """

    @property
    @abstractmethod
    def time(self) -> float:
        """Current simulation time."""
        ...

    @abstractmethod
    def rates(self) -> NDArray[np.float64]:
        """Compute all event rates given the current state.

        Returns:
            1-D array of non-negative rates. Length and interpretation
            are process-specific. The Gillespie engine selects event ``i``
            with probability ``rates[i] / rates.sum()``.
        """
        ...

    @abstractmethod
    def execute(self, event_index: int, dt: float) -> None:
        """Apply the selected event and advance time by ``dt``.

        Args:
            event_index: Index into the array returned by ``rates()``.
            dt: Time elapsed since the previous event.
        """
        ...

    @abstractmethod
    def is_absorbing(self) -> bool:
        """Return True if no further events can occur.

        The engine checks this *before* computing rates, so returning
        True here guarantees ``rates()`` will not be called on a dead state.
        """
        ...

    @abstractmethod
    def scalar_output(self) -> float:
        """Scalar summary for convergence monitoring.

        The engine feeds this value into a Welford running-mean tracker
        after each realisation. Choose the quantity whose relative SEM
        you want to converge (e.g. final epidemic size, total event count).
        """
        ...

    def should_accept(self) -> bool:
        """Whether this realisation should be included in ensemble statistics.

        Override to implement filtering (e.g. discard sub-critical epidemics).
        Discarded runs still count toward ``max_realizations`` but not
        toward ``min_realizations`` or convergence.

        Default: always accept.
        """
        return True

    @abstractmethod
    def trajectory(self) -> Trajectory:
        """Extract the trajectory recorded during simulation.

        Called once after the simulation loop terminates. The returned
        ``Trajectory`` is frozen — safe to store, aggregate, or serialise.
        """
        ...


class ProcessFactory(ABC):
    """Creates fresh process instances for each realisation.

    The Gillespie engine calls ``create()`` once per realisation rather
    than resetting state, ensuring clean isolation between runs.

    Implementations hold the immutable configuration and network topology;
    ``create()`` builds a new mutable process from them.
    """

    @abstractmethod
    def create(self, rng: np.random.Generator) -> ContinuousTimeProcess:
        """Build a fresh process instance with the given RNG.

        Args:
            rng: Random number generator for this realisation.
                 Each realisation should receive an independent generator.
        """
        ...

    @property
    @abstractmethod
    def convergence_config(self) -> ConvergenceConfig:
        """Convergence parameters for the ensemble."""
        ...
