# craeft

**Random graphs. Controlled structure. Stochastic simulation.**

craeft is a Python library that generates random graphs with an exact
degree sequence, designed counts of chosen subgraphs and a clustering 
coefficient computable in closed form before the
build begins.

Graphs like this make controlled experiments possible. Two families
can have eqaul summary statistics: degrees, edge counts,
clustering but still differ in their subgraph arrangements. craeft
builds families that differ in exactly this way.

Alongside the configuration-model generator, the library ships Erdős–Rényi
and degree-preserving rewiring baselines. The simulation engine accepts any
process implementing the `ContinuousTimeProcess` interface, with convergence
monitoring, ensemble aggregation and multiprocessing; SIR epidemics are
included as the reference implementation.

## Bringing the Claims into Focus

The claims below are stated precisely, measured, and shipped with the scripts
that produced every number — see [The Claims](claims/index.md).

1. **[The degree sequence is exact](claims/claim-1-exact-degrees.md)** —
   element-wise, on every build, asserted at runtime.
2. **[Subgraph counts are set by the input](claims/claim-2-designed-counts.md)** —
   realized count = designed count + a floor that is predictable in closed
   form.
3. **[Clustering is known before the graph exists](claims/claim-3-clustering-in-advance.md)** —
   a closed-form calculation from the configuration alone.
4. **[Higher-order structure stays free](claims/claim-4-residual-freedom.md)** —
   and what stays free is measured, not ignored.

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

# Generate a clustered graph with triangle subgraphs
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

## Where next

- **[Getting started](getting-started.md)** — from installation to a first
  graph and simulation.
- **[Concepts](concepts/ccm.md)** — the clustered configuration model, the
  sampling scheme behind it, and the Gillespie algorithm.
- **[The Claims](claims/index.md)** — what is guaranteed, what is assumed,
  and what is left uncontrolled, with the audit scripts behind every number.
- **[API reference](api/index.md)** — generators, metrics, the simulation
  engine and plotting, documented from the source.
