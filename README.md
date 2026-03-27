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
| `NetworkGenerator` | Protocol for network generation — `generate(rng) -> csr_matrix` |
| `ContinuousTimeProcess` | ABC for any CTMC — defines `rates()`, `execute()`, `trajectory()` |
| `ProcessFactory` | ABC for creating fresh process instances per realisation |
| `EpidemicSimulator` | Protocol for running ensemble simulations on a network |

All generators are frozen dataclasses (stateless, picklable) and all
simulation is driven through the generic Gillespie engine.

## Quick start

```python
import numpy as np
from craeft import configuration_model, global_clustering_coefficient

rng = np.random.default_rng(42)
degrees = np.full(500, 5)

# Unclustered
adj = configuration_model(degrees, rng=rng)
print(global_clustering_coefficient(adj))  # ~0.0

# With clustering
adj = configuration_model(degrees, phi=0.2, rng=rng)
print(global_clustering_coefficient(adj))  # ~0.2
```

## Network generators

| Generator | What it does |
|-----------|-------------|
| `configuration_model(degrees, phi)` | Standard or clustered configuration model |
| `random_graph(n, p)` | Erdos-Renyi G(n, p) |
| `big_v_rewire(adj, target_clustering)` | Degree-preserving rewiring to increase clustering |
| `motif_decomposition(n, clique_size, target_clustering)` | Start from cliques, rewire down to target |

All return scipy `csr_matrix` adjacency matrices.

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
