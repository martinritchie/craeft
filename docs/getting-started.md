# Getting Started

This guide walks through the core workflows: generating networks with
controlled clustering and running epidemic simulations on them.

## Generating networks

### Erdos-Renyi random graph

The simplest model — each edge exists independently with probability $p$.

```python
import numpy as np
from craeft.graphs.erdos_renyi import ErdosRenyiGraph, ErdosRenyiConfig

rng = np.random.default_rng(42)
config = ErdosRenyiConfig(n=1000, p=0.01)
graph = ErdosRenyiGraph.from_config(config, rng)
adjacency = graph.to_csr()
```

### Configuration model (unclustered)

Prescribe an exact degree sequence. The result has near-zero clustering.

```python
from craeft.graphs.configuration_model import ConfigModelConfig, ConfigModelGraph

degrees = np.full(1000, 5)  # all nodes degree 5
config = ConfigModelConfig(n=1000, degrees=degrees)
graph = ConfigModelGraph.from_config(config, rng)
print(f"Clustering: {graph.clustering_coefficient:.4f}")
```

### Clustered configuration model

Embed subgraph structures to produce non-zero clustering. Each
`SubgraphSequence` pairs a subgraph with a discrete distribution
controlling how many times each node participates in that subgraph.

```python
from scipy.stats import poisson
from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model.sequence import SubgraphSequence

# Define subgraph structures
triangle = Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
diamond = Subgraph(adjacency=np.array([
    [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]
]))

# Pair with participation distributions
tri_seq = SubgraphSequence(subgraph=triangle, distribution=poisson(0.2))
dia_seq = SubgraphSequence(subgraph=diamond, distribution=poisson(0.1))

config = ConfigModelConfig(
    n=1000,
    degrees=degrees,
    sequences=(tri_seq, dia_seq),
    max_retries=100,
)
graph = ConfigModelGraph.from_config(config, rng)
print(f"Clustering: {graph.clustering_coefficient:.4f}")
```

The pipeline automatically handles orbit decomposition for non-vertex-transitive
subgraphs (like the diamond, which has hub and leaf orbits).

### Using generator objects (legacy API)

For experiments that generate many networks with the same parameters,
the legacy `networks/` API provides frozen dataclass generators:

```python
from craeft.networks.generation.generator import (
    PoissonNetworkGenerator,
    BigVRewiringGenerator,
    MotifDecompositionGenerator,
)

# Poisson degree distribution with clustering
gen = PoissonNetworkGenerator(n=1000, mean_degree=5, max_degree=20, phi=0.2)

# Big-V rewiring on top of an unclustered base
base = PoissonNetworkGenerator(n=1000, mean_degree=5, max_degree=20)
gen = BigVRewiringGenerator(base=base, target_clustering=0.2)

adjacency = gen.generate(rng)
```

## Measuring network structure

```python
from craeft.graphs.metrics.clustering import global_clustering_coefficient

print(f"Connected: {graph.is_connected}")
print(f"Clustering: {graph.clustering_coefficient:.4f}")
print(f"Degrees: {graph.degrees}")
```

The graph objects expose properties directly. For raw adjacency matrices,
use the standalone metrics functions.

## Running SIR epidemics

### Single realisation

Use the low-level `run_once` function for a single Gillespie trajectory.

```python
from craeft.point_processes import run_once, ConvergenceConfig
from craeft.point_processes.epidemics import SIRConfig, SIRProcessFactory

sir = SIRConfig(tau=1.0, gamma=1.0, initial_infected=5)
convergence = ConvergenceConfig(t_end=15.0)
factory = SIRProcessFactory(
    config=sir, adjacency=adjacency, convergence_config=convergence
)

trajectory, final_size, accepted = run_once(factory, t_end=15.0, rng=rng)
print(f"Final size: {final_size}")
```

### Ensemble with convergence

The `SIRSimulator` runs multiple realisations, monitors convergence of
the mean final epidemic size via the relative standard error, and
optionally filters sub-critical outbreaks.

```python
from craeft.point_processes.epidemics import SIRSimulator

convergence = ConvergenceConfig(
    t_end=15.0,
    convergence_threshold=0.05,
    min_realizations=30,
    max_realizations=500,
)
simulator = SIRSimulator(config=sir, convergence=convergence)
result = simulator.run(adjacency, rng)

print(f"Converged: {result.convergence.converged}")
print(f"Realisations: {result.convergence.n_realizations}")
print(f"Mean final size: {result.scalar_output_mean:.1f}")
```

## Running full experiments

The `Experiment` class composes a generator and simulator, running
multiple network realisations with optional parallelism.

```python
from craeft import Experiment
from craeft.networks.generation.generator import PoissonNetworkGenerator
from craeft.point_processes.epidemics import SIRSimulator, SIRConfig
from craeft.point_processes import ConvergenceConfig

generator = PoissonNetworkGenerator(
    n=500, mean_degree=5, max_degree=20, phi=0.2
)
simulator = SIRSimulator(
    config=SIRConfig(tau=1.0, gamma=1.0, initial_infected=5),
    convergence=ConvergenceConfig(t_end=15.0, max_realizations=200),
)

experiment = Experiment(generator=generator, simulator=simulator, n_networks=10)
results = experiment.run(rng=rng, n_workers=1)

for i, ensemble in enumerate(results):
    print(f"Network {i}: final size = {ensemble.scalar_output_mean:.1f}")
```

## Plotting

```python
from craeft.utils.plotting import plot_sir, plot_prevalence_comparison

fig = plot_sir(
    t=result.t,
    s_mean=result.means["susceptible"],
    i_mean=result.means["infected"],
    r_mean=result.means["recovered"],
    s_std=result.stds["susceptible"],
    i_std=result.stds["infected"],
    r_std=result.stds["recovered"],
)
fig.savefig("sir_dynamics.pdf")
```