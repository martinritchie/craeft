# craeft

**Random networks. Controlled structure. Stochastic simulation.**

craeft provides tools for constructing random networks where you control
not just the degree sequence but also the clustering coefficient and
local motif structure. It pairs these generators with a generic Gillespie
simulation engine for running any continuous-time Markov chain on the
resulting networks.

## Features

- **Network generation with controlled clustering** — Erdos-Renyi,
  configuration model, clustered configuration model, and degree-preserving
  rewiring algorithms
- **Motif-aware construction** — embed triangles, cliques, and other
  substructures as first-class building blocks using Pržulj graphlet notation
- **Multiple generation strategies** — choose between the clustered
  configuration model (CCM), Big-V rewiring, or motif decomposition
  depending on your needs
- **Process-agnostic simulation engine** — Gillespie algorithm with
  convergence monitoring, ensemble aggregation, and multiprocessing support
- **Extensible point processes** — implement the `ContinuousTimeProcess`
  interface for any CTMC; SIR epidemics included as a reference implementation

## Quick example

```python
import numpy as np
from craeft import (
    configuration_model,
    global_clustering_coefficient,
    is_connected,
)
from craeft.point_processes.epidemics import SIRConfig, SIRSimulator
from craeft.point_processes import ConvergenceConfig

rng = np.random.default_rng(42)

# Generate a clustered network (degree-5, phi=0.2)
degrees = np.full(500, 5)
adjacency = configuration_model(degrees, phi=0.2, rng=rng)

print(f"Connected: {is_connected(adjacency)}")
print(f"Clustering: {global_clustering_coefficient(adjacency):.3f}")

# Run an SIR epidemic ensemble
sir = SIRConfig(tau=1.0, gamma=1.0, initial_infected=5)
convergence = ConvergenceConfig(
    t_end=15.0,
    min_realizations=50,
    max_realizations=500,
)
simulator = SIRSimulator(config=sir, convergence=convergence)
result = simulator.run(adjacency, rng)

print(f"Mean final size: {result.scalar_output_mean:.1f}")
```

## Installation

```bash
uv add craeft
```

With plotting support:

```bash
uv add "craeft[plot]"
```

## Project layout

```
src/craeft/
├── networks/
│   ├── generation/          # Network generators
│   │   ├── configuration_model/  # CCM pipeline
│   │   ├── distributions/   # Degree distributions
│   │   └── motifs/          # Graphlet definitions
│   ├── metrics/             # Clustering, connectivity
│   └── rewiring/            # Big-V, motif decomposition
├── point_processes/
│   ├── epidemics/           # SIR model
│   ├── gillespie.py         # Simulation engine
│   └── process.py           # Core abstractions
├── utils/
│   └── plotting.py          # Visualisation
└── experiment.py            # Orchestration
```
