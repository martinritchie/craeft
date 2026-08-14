# craeft

**Random networks. Controlled structure. Stochastic simulation.**

craeft provides tools for constructing random networks where you control
not just the degree sequence but also the clustering coefficient and
local motif structure. It pairs these generators with a generic Gillespie
simulation engine for running any continuous-time Markov chain on the
resulting networks.

## Features

- **Network generation with controlled clustering** — configuration model
  with orbit-aware subgraph embedding, Erdos-Renyi, and degree-preserving
  rewiring algorithms
- **Orbit-aware motif construction** — embed triangles, diamonds, cliques, and
  other substructures with correct handling of non-vertex-transitive subgraphs
  (via sequential conditional sampling)
- **Config → graph pipeline** — stateless `GraphConfig` dataclasses produce
  `BaseGraph` objects with CSR-backed adjacency, cluster coefficient, degrees,
  and export methods
- **Process-agnostic simulation engine** — Gillespie algorithm with
  convergence monitoring, ensemble aggregation, and multiprocessing support
- **Extensible point processes** — implement the `ContinuousTimeProcess`
  interface for any CTMC; SIR epidemics included as a reference implementation

## Quick example

```python
import numpy as np
from scipy.stats import poisson

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model import ConfigModelConfig, ConfigModelGraph
from craeft.graphs.configuration_model.sequence import SubgraphSequence
from craeft.point_processes.epidemics import SIRConfig, SIRSimulator
from craeft.point_processes import ConvergenceConfig

rng = np.random.default_rng(42)

# Generate a clustered network with triangle subgraphs
triangle = Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
tri_seq = SubgraphSequence(subgraph=triangle, distribution=poisson(0.2))

degrees = np.full(500, 5)
config = ConfigModelConfig(n=500, degrees=degrees, sequences=(tri_seq,))
graph = ConfigModelGraph.from_config(config, rng)

print(f"Connected: {graph.is_connected}")
print(f"Clustering: {graph.clustering_coefficient:.3f}")

# Run an SIR epidemic ensemble on the resulting adjacency
sir = SIRConfig(tau=1.0, gamma=1.0, initial_infected=5)
convergence = ConvergenceConfig(
    t_end=15.0,
    min_realizations=50,
    max_realizations=500,
)
simulator = SIRSimulator(config=sir, convergence=convergence)
result = simulator.run(graph.to_csr(), rng)

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
├── graphs/                    # Graph generation (new architecture)
│   ├── base.py                # BaseGraph, UndirectedGraph, Subgraph
│   ├── erdos_renyi.py         # Erdos-Renyi G(n, p)
│   ├── metrics/               # Clustering, connectivity
│   └── configuration_model/   # Config model + CMA pipeline
│       ├── models.py          # ConfigModelConfig, ConfigModelGraph
│       ├── connection.py      # Connector (subgraph + single pairing)
│       └── sequence/          # Participation sampling, orbit splitting, allocation
├── networks/                  # Network generation (legacy API)
│   ├── generation/
│   │   ├── configuration_model/  # Legacy CCM pipeline
│   │   ├── distributions/    # Degree distributions
│   │   └── motifs/           # Graphlet definitions (G0–G29)
│   ├── metrics/              # Clustering, connectivity
│   └── rewiring/             # Big-V, motif decomposition
├── point_processes/
│   ├── epidemics/            # SIR model
│   ├── gillespie.py          # Simulation engine
│   └── process.py            # Core abstractions
├── utils/
│   └── plotting.py           # Visualisation
└── experiment.py             # Orchestration
```