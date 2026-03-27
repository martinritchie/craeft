"""SIR epidemic as a continuous-time point process.

Implements the ``ContinuousTimeProcess`` interface for the
Susceptible-Infected-Recovered model on a static network.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Field, computed_field
from scipy.sparse import csr_array

from craeft.point_processes.process import (
    ContinuousTimeProcess,
    ConvergenceConfig,
    ProcessFactory,
    Trajectory,
)

# Agent states
S, I, R = 0, 1, 2  # noqa: E741


class SIRConfig(BaseModel):
    """SIR epidemic model configuration.

    Contains only SIR-specific parameters. Convergence and ensemble
    parameters live in ``ConvergenceConfig``.

    Example:
        >>> config = SIRConfig(tau=2.0, gamma=0.5)
        >>> config.R0
        4.0
    """

    tau: float = Field(
        default=1.0,
        gt=0,
        description="Transmission rate per S-I edge per unit time",
    )

    gamma: float = Field(
        default=1.0,
        gt=0,
        description="Recovery rate per infected per unit time",
    )

    initial_infected: int = Field(
        default=1,
        ge=1,
        description="Number of initially infected nodes",
    )

    filter_subcritical: bool = Field(
        default=False,
        description="Discard sub-critical runs where final_size <= initial_infected",
    )

    method: Literal["ode", "stochastic", "both"] = Field(
        default="both",
        description="Simulation method",
    )

    @computed_field
    @property
    def R0(self) -> float:
        """Basic reproduction number (for homogeneous mixing)."""
        return self.tau / self.gamma


class SIRProcess(ContinuousTimeProcess):
    """SIR epidemic dynamics on a static network.

    All SIR-specific logic (infection pressure, recovery rates, S->I->R
    transitions) lives here. The Gillespie engine sees only the
    ``ContinuousTimeProcess`` interface.

    Example:
        >>> from craeft.point_processes.gillespie import run_once
        >>> from craeft.point_processes.epidemics.sir import (
        ...     SIRConfig, SIRProcessFactory,
        ... )
        >>> config = SIRConfig(tau=0.5, gamma=1.0)
        >>> factory = SIRProcessFactory(
        ...     config=config, adjacency=adj, convergence_config=cc,
        ... )
        >>> trajectory, scalar, accepted = run_once(
        ...     factory, t_end=20.0, rng=rng,
        ... )
    """

    def __init__(
        self,
        config: SIRConfig,
        adjacency: csr_array,
    ) -> None:
        self._config = config
        self._adjacency = adjacency

        n = adjacency.shape[0]
        self._n = n
        self._agent_state = np.zeros(n, dtype=np.int8)
        self._agent_state[: config.initial_infected] = I

        # Trajectory accumulators
        self._t: list[float] = [0.0]
        self._s: list[int] = [n - config.initial_infected]
        self._i: list[int] = [config.initial_infected]

    # -- ContinuousTimeProcess interface --

    @property
    def time(self) -> float:
        return self._t[-1]

    def rates(self) -> NDArray[np.float64]:
        """Concatenated ``[infection_rates..., recovery_rates...]``.

        Event index ``< n``  -> infection of node ``index``.
        Event index ``>= n`` -> recovery of node ``index - n``.
        """
        infection = self._infection_rates()
        recovery = self._recovery_rates()
        return np.concatenate([infection, recovery])

    def execute(self, event_index: int, dt: float) -> None:
        node = event_index % self._n
        self._t.append(self._t[-1] + dt)
        self._agent_state[node] += 1  # S->I or I->R

        if event_index < self._n:
            self._s.append(self._s[-1] - 1)
            self._i.append(self._i[-1] + 1)
        else:
            self._s.append(self._s[-1])
            self._i.append(self._i[-1] - 1)

    def is_absorbing(self) -> bool:
        return self._i[-1] == 0

    def scalar_output(self) -> float:
        """Final epidemic size (total recovered)."""
        return float(self._n - self._s[-1] - self._i[-1])

    def should_accept(self) -> bool:
        if not self._config.filter_subcritical:
            return True
        return self.scalar_output() > self._config.initial_infected

    def trajectory(self) -> Trajectory:
        t = np.array(self._t)
        s = np.array(self._s)
        i = np.array(self._i)
        r = self._n - s - i
        return Trajectory(
            t=t,
            compartments={"s": s, "i": i, "r": r},
        )

    # -- SIR-specific rate computation --

    def _infection_rates(self) -> NDArray[np.float64]:
        """Per-node infection rate: tau * (number of infected neighbours)."""
        infected_mask = (self._agent_state == I).astype(np.float64)
        infected_neighbours = self._adjacency @ infected_mask
        susceptible_mask = self._agent_state == S
        return np.where(susceptible_mask, self._config.tau * infected_neighbours, 0.0)

    def _recovery_rates(self) -> NDArray[np.float64]:
        """Per-node recovery rate: gamma if infected, else 0."""
        infected_mask = self._agent_state == I
        return np.where(infected_mask, self._config.gamma, 0.0)


@dataclass(frozen=True)
class SIRProcessFactory(ProcessFactory):
    """Creates fresh ``SIRProcess`` instances for each realisation.

    Holds the immutable SIR config and adjacency matrix. The Gillespie
    engine calls ``create()`` once per run.
    """

    config: SIRConfig
    adjacency: csr_array
    _convergence_config: ConvergenceConfig

    def create(self, rng: np.random.Generator) -> SIRProcess:
        return SIRProcess(self.config, self.adjacency)

    @property
    def convergence_config(self) -> ConvergenceConfig:
        return self._convergence_config
