# Clustered Configuration Model

The Clustered Configuration Model (CCM) generates networks with both a
prescribed degree sequence *and* a target clustering coefficient $\phi$. It
does this by embedding subgraph instances (motifs) into the network, then
pairing remaining single stubs via the standard configuration model.

!!! note "Reference"
    The core algorithm is described in: Ritchie, M., Berthouze, L., & Kiss,
    I. Z. (2014). Higher-order structure and epidemic dynamics in clustered
    networks. *Journal of Mathematical Biology*, 72(3), 483–511.

    The orbital decomposition for non-vertex-transitive subgraphs is a generalisation
    of the original paper's uniform-cardinality assumption, described in
    `sequential-conditional-sampling.md`.

## Architecture

The configuration model is exposed through a **config → graph** pipeline:

```python
ConfigModelConfig(n, degrees, sequences) → ConfigModelGraph.from_config() → CSR adjacency
```

This replaces the older `configuration_model(degrees, phi)` function API with
a two-tier architecture:

- **Vanilla CM** — when `sequences` is empty, pairs stubs from the degree sequence
- **Clustered CM** — when `sequences` contains one or more `SubgraphSequence`
  entries, embeds subgraph instances before pairing remaining singles

## Algorithm pipeline

### Vanilla configuration model (no sequences)

1. Shuffle and pair stubs from the degree sequence
2. Skip self-loops and multi-edges
3. Assemble into a symmetric adjacency matrix

### Clustered model (with subgraph sequences)

1. **Sample** — for each `SubgraphSequence`, draw a participation sequence from
   the specified distribution (e.g. `poisson(0.3)`)
2. **Split by orbit** — for non-vertex-transitive subgraphs (e.g. diamond, bowtie),
   assign each node's participations to specific orbit roles using sequential
   conditional sampling from a shared urn. Vertex-transitive subgraphs (triangle,
   square, K4) skip this step.
3. **Allocate** — verify that every node's total subgraph stub cost (sum over
   orbits of count × orbit_degree) fits within its degree budget
4. **Connect subgraphs** — for each subgraph type, draw concrete nodes from
   per-orbit pools, check for duplicate nodes and existing edges, and form the
   subgraph's internal edges
5. **Pair singles** — pair remaining single stubs, avoiding edges already formed
   by subgraphs
6. **Assemble** — combine all edges into a symmetric CSR adjacency matrix

On any failure (degree budget exceeded, unresolvable collisions), the pipeline
retries from step 1 with a fresh sample, up to `max_retries` times.

## Orbit-aware decomposition

A key contribution over the original paper is support for **non-vertex-transitive
subgraphs** — subgraphs where not all vertices are equivalent under automorphism.

| Subgraph | Orbits | Transitive? |
|----------|--------|-------------|
| Triangle (G2), Square (G5), K4 (G8), K5 (G29) | 1 | Yes |
| Diamond (G7) | 2 (hubs, leaves) | No |
| Bowtie (G14), House (G17) | 2–3 | No |

For vertex-transitive subgraphs all vertices are equivalent, so the split step
is a no-op. For non-transitive subgraphs, the sequential conditional sampling
algorithm ensures exact orbit proportions without global rejection. See
`sequential-conditional-sampling.md` for the full algorithm.

## Usage

```python
import numpy as np
from scipy.stats import poisson

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model import ConfigModelConfig, ConfigModelGraph
from craeft.graphs.configuration_model.sequence import SubgraphSequence

rng = np.random.default_rng(42)

# --- Vanilla configuration model ---
degrees = np.full(500, 5)
config = ConfigModelConfig(n=500, degrees=degrees)
graph = ConfigModelGraph.from_config(config, rng)
print(f"Vanilla CM clustering: {graph.clustering_coefficient:.4f}")

# --- Clustered model with triangles ---
triangle = Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
tri_seq = SubgraphSequence(subgraph=triangle, distribution=poisson(0.2))

config = ConfigModelConfig(
    n=500,
    degrees=degrees,
    sequences=(tri_seq,),
    max_retries=100,
)
graph = ConfigModelGraph.from_config(config, rng)
print(f"Clustered CM clustering: {graph.clustering_coefficient:.4f}")

# --- Multiple subgraph types ---
diamond = Subgraph(adjacency=np.array([[0,1,1,1],[1,0,1,0],[1,1,0,1],[1,0,1,0]]))
diamond_seq = SubgraphSequence(subgraph=diamond, distribution=poisson(0.1))

config = ConfigModelConfig(
    n=500,
    degrees=degrees,
    sequences=(tri_seq, diamond_seq),
    max_retries=200,
)
graph = ConfigModelGraph.from_config(config, rng)
print(f"Triangle+diamond clustering: {graph.clustering_coefficient:.4f}")

# Access graph properties
print(f"Edges: {graph.n_edges}")
print(f"Degrees: {graph.degrees}")
adj = graph.to_csr()  # scipy sparse CSR matrix
```

## Retry behaviour

The subgraph pipeline uses rejection sampling at multiple levels:

1. **Participation sampling** — the distribution may produce values incompatible
   with the degree budget (e.g. a node told to participate 5× in triangles with
   only degree 4). Guarded by `max_iterations` in `_sample_sequence`.
2. **Orbit splitting** — the sequential urn algorithm detects infeasible splits
   (e.g. insufficient hub slots remaining). Raises `ValueError`.
3. **Allocation** — the degree budget check catches overall excess. Raises
   `AllocationError`.
4. **Connection** — subgraph connection detects unresolvable collisions (duplicate
   nodes, existing edges). Raises `ConnectionError`.

All are caught by the outer retry loop in `ConfigModelGraph.from_config`, which
resamples participation sequences and retries up to `max_retries` times.