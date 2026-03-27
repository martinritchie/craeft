# Gillespie Algorithm

The Gillespie algorithm (also known as the Stochastic Simulation Algorithm)
is an exact method for simulating continuous-time Markov chains (CTMCs).
craeft uses it as the engine for all stochastic point process simulations.

!!! info "Reference"
    Gillespie, D. T. (1977). Exact stochastic simulation of coupled chemical
    reactions. *The Journal of Physical Chemistry*, 81(25), 2340–2361.

## How it works

At each step:

1. **Compute rates** — each possible event has a rate $r_i \geq 0$
2. **Sample waiting time** — $\Delta t \sim \text{Exp}(R)$ where
   $R = \sum_i r_i$ is the total rate
3. **Select event** — event $j$ is chosen with probability $r_j / R$
4. **Execute** — update the system state and advance the clock by $\Delta t$
5. **Repeat** until the process reaches an absorbing state or the time
   limit is reached

The algorithm is *exact* — it produces trajectories from the true
distribution of the CTMC without time-discretisation error.

## Process-agnostic design

craeft decouples the Gillespie engine from the specific dynamics being
simulated. The engine only needs an object implementing the
`ContinuousTimeProcess` interface:

```python
class ContinuousTimeProcess(ABC):
    @abstractmethod
    def rates(self) -> NDArray[np.float64]: ...

    @abstractmethod
    def execute(self, event_index: int, dt: float) -> None: ...

    @abstractmethod
    def is_absorbing(self) -> bool: ...

    @abstractmethod
    def trajectory(self) -> Trajectory: ...

    @abstractmethod
    def scalar_output(self) -> float: ...
```

This means you can implement any CTMC (SIR, SIS, Hawkes processes, etc.)
and plug it into the same simulation infrastructure.

## Ensemble simulation

Running a single realisation of a stochastic process tells you little.
The `simulate` function runs many realisations and aggregates results:

- **Convergence monitoring** — uses Welford's online algorithm to track
  the running mean and variance of a scalar output (e.g., final epidemic
  size). Stops when the relative standard error drops below a threshold.
- **Sub-critical filtering** — optionally discards realisations where
  the process dies out quickly (controlled by `should_accept()`).
- **Trajectory aggregation** — interpolates all accepted trajectories
  onto a common time grid and computes pointwise mean and standard
  deviation for each compartment.
- **Parallelism** — distributes realisations across workers via
  `ProcessPoolExecutor` when `num_workers > 1`.

## SIR as a concrete example

The SIR (Susceptible-Infected-Recovered) model on a network is the
primary concrete implementation:

- **Events**: infection (S → I) and recovery (I → R)
- **Infection rate** for node $i$: $\tau \times |\{j \in \text{neighbours}(i) : j \text{ is infected}\}|$
- **Recovery rate** for node $i$: $\gamma$ (constant per infected node)
- **Scalar output**: final epidemic size (total recovered)

The rates vector has length $2n$ — infection rates for nodes $0 \ldots n{-}1$,
recovery rates for nodes $n \ldots 2n{-}1$. Infection pressure is
computed via a sparse matrix-vector multiply: $\mathbf{A} \cdot \mathbf{x}$
where $\mathbf{x}$ is the indicator vector of infected nodes.
