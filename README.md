# craeft

[![Docs](https://img.shields.io/badge/docs-GitHub%20Pages-blue)](https://martinritchie.github.io/craeft/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Random networks. Controlled structure. Stochastic simulation.**

craeft (pronounced "craft") provides modern network generation algorithms
and the option to simulate point processes on them.

## Install

```bash
uv add craeft
```

## Core interfaces

| Interface | Description |
|-----------|-------------|
| `GraphConfig` / `BaseGraph` | Config → graph pipeline with CSR-backed adjacency |
| `SubgraphSequence` | Subgraph + participation distribution for clustered models |
| `ContinuousTimeProcess` | ABC for any CTMC — defines `rates()`, `execute()`, `trajectory()` |
| `EpidemicSimulator` | Protocol for running ensemble simulations on a network |

## Quick start

```python
import numpy as np
from scipy.stats import poisson
from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model import ConfigModelConfig, ConfigModelGraph
from craeft.graphs.configuration_model.sequence import SubgraphSequence

rng = np.random.default_rng(42)
degrees = np.full(500, 5)

# Vanilla configuration model (no subgraphs → near-zero clustering)
config = ConfigModelConfig(n=500, degrees=degrees)
graph = ConfigModelGraph.from_config(config, rng)
print(graph.clustering_coefficient)  # ~0.0

# Clustered model with triangle subgraphs
triangle = Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
tri_seq = SubgraphSequence(subgraph=triangle, distribution=poisson(0.2))
config = ConfigModelConfig(n=500, degrees=degrees, sequences=(tri_seq,))
graph = ConfigModelGraph.from_config(config, rng)
print(graph.clustering_coefficient)  # > 0.0

adj = graph.to_csr()  # scipy sparse CSR matrix
```

## Network generators

| Generator | What it does |
|-----------|-------------|
| `ConfigModelGraph.from_config()` | Standard or clustered configuration model with orbit-aware subgraph embedding |
| `ErdosRenyiGraph.from_config()` | Erdos-Renyi G(n, p) |
| `big_v_rewire(adj, target_clustering)` | Degree-preserving rewiring to increase clustering |
| `motif_decomposition(n, clique_size, target_clustering)` | Start from cliques, rewire down to target |

## Simulation

The Gillespie engine is process-agnostic. SIR epidemics are included as
a reference implementation:

```python
from craeft.point_processes.epidemics import SIRConfig, SIRSimulator
from craeft.point_processes import ConvergenceConfig

simulator = SIRSimulator(
    config=SIRConfig(tau=1.0, gamma=1.0, initial_infected=5),
    convergence=ConvergenceConfig(t_end=15.0, max_realizations=500),
)
result = simulator.run(adj, rng)
print(f"Mean final size: {result.scalar_output_mean:.0f}")
```

Implement `ContinuousTimeProcess` to plug in your own dynamics.

## Documentation

```bash
uv sync
uv run mkdocs serve
```

## License

MIT