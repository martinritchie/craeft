# Getting Started

This guide walks through the core workflows: generating networks with
controlled clustering and running epidemic simulations on them.

## Generating networks

### Erdos-Renyi random graph

The simplest model — each edge exists independently with probability $p$.

```python
import numpy as np
from craeft import random_graph

rng = np.random.default_rng(42)
adjacency = random_graph(n=1000, p=0.01, rng=rng)
```

### Configuration model (unclustered)

Prescribe an exact degree sequence. The result has near-zero clustering.

```python
from craeft import configuration_model

degrees = np.full(1000, 5)  # all nodes degree 5
adjacency = configuration_model(degrees, rng=rng)
```

### Clustered configuration model

Add clustering by setting `phi > 0`. The CCM allocates a fraction of each
node's stubs to motif participation (triangles, K4 cliques) before pairing
the remainder via the standard configuration model.

```python
adjacency = configuration_model(degrees, phi=0.2, rng=rng)
```

### Using generator objects

For experiments that generate many networks with the same parameters,
use the generator dataclasses. These are frozen, picklable, and compose
cleanly with the `Experiment` orchestrator.

```python
from craeft import (
    PoissonNetworkGenerator,
    BigVRewiringGenerator,
    MotifDecompositionGenerator,
)

# Poisson degree distribution with clustering
gen = PoissonNetworkGenerator(n=1000, mean_degree=5, max_degree=20, phi=0.2)

# Big-V rewiring on top of an unclustered base
base = PoissonNetworkGenerator(n=1000, mean_degree=5, max_degree=20)
gen = BigVRewiringGenerator(base=base, target_clustering=0.2)

# Motif decomposition (start from cliques, tear down to target)
gen = MotifDecompositionGenerator(
    num_nodes=1000, clique_size=6, target_clustering=0.2,
)

adjacency = gen.generate(rng)
```

## Measuring network structure

```python
from craeft import global_clustering_coefficient, is_connected
from craeft.networks.metrics import (
    count_triangles,
    local_clustering,
    triangles_per_node,
)

print(f"Connected: {is_connected(adjacency)}")
print(f"Triangles: {count_triangles(adjacency)}")
print(f"Global clustering: {global_clustering_coefficient(adjacency):.4f}")
```

## Running SIR epidemics

### Single realisation

Use the low-level `run_once` function for a single Gillespie trajectory.

```python
from craeft.point_processes import run_once, ConvergenceConfig
from craeft.point_processes.epidemics import SIRConfig, SIRProcessFactory

sir = SIRConfig(tau=1.0, gamma=1.0, initial_infected=5)
convergence = ConvergenceConfig(t_end=15.0)
factory = SIRProcessFactory(config=sir, adjacency=adjacency, convergence_config=convergence)

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
print(f"Mean final size: {result.scalar_output_mean:.1f} ± {result.scalar_output_std:.1f}")
```

## Running full experiments

The `Experiment` class composes a generator and simulator, running
multiple network realisations with optional parallelism.

```python
from craeft import Experiment, PoissonNetworkGenerator
from craeft.point_processes.epidemics import SIRSimulator, SIRConfig
from craeft.point_processes import ConvergenceConfig

generator = PoissonNetworkGenerator(n=500, mean_degree=5, max_degree=20, phi=0.2)
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
