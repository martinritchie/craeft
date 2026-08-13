# Motifs & Notation

Motifs are small, recurring substructures used as building blocks in the
Clustered Configuration Model. craeft adopts the **Pržulj graphlet
notation** (G0–G29) as the canonical naming scheme.

!!! info "Reference"
    Pržulj, N., Corneil, D. G., & Jurisica, I. (2004). Modeling
    interactome: Scale-free or geometric? *Bioinformatics*, 20(18),
    3508–3515.

## Graphlet catalogue

### 2–3 node graphlets

| ID | Structure | Nodes | Edges | Transitive |
|----|-----------|-------|-------|------------|
| **G0** | Single edge (K₂) | 2 | 1 | yes |
| **G1** | Path P₂ (wedge) | 3 | 2 | no |
| **G2** | Complete K₃ (triangle) | 3 | 3 | yes |

### 4-node graphlets

| ID | Structure | Edges | Transitive | Orbits | Hamiltonian |
|----|-----------|-------|------------|--------|-------------|
| **G3** | Path P₃ | 3 | no | 2 | no |
| **G4** | Star S₃ (claw) | 3 | no | 2 | no |
| **G5** | Cycle C₄ (square) | 4 | yes | 1 | yes |
| **G6** | Triangle + pendant (paw) | 4 | no | 3 | no |
| **G7** | K₄ − e (diamond) | 5 | no | 2 | yes |
| **G8** | Complete K₄ | 6 | yes | 1 | yes |

### 5-node graphlets (selected)

| ID | Structure | Edges | Transitive | Hamiltonian |
|----|-----------|-------|------------|-------------|
| **G12** | Cycle C₅ (pentagon) | 5 | yes | yes |
| **G14** | Two triangles sharing edge (bowtie) | 6 | no | yes |
| **G17** | Square + triangle (house) | 6 | no | yes |
| **G29** | Complete K₅ | 10 | yes | yes |

### Extended motifs (6+ nodes)

| ID | Structure | Nodes | Edges | Transitive | Hamiltonian |
|----|-----------|-------|-------|------------|-------------|
| **C6** | Cycle C₆ (hexagon) | 6 | 6 | yes | yes |
| **K6** | Complete K₆ | 6 | 15 | yes | yes |

## CCM compatibility

The CCM requires motifs with a **Hamiltonian cycle** (a cycle visiting
every node exactly once). The new `graphs/` architecture supports all
Hamiltonian-connected motifs, including non-vertex-transitive ones.

For non-transitive subgraphs (like the diamond G7 or bowtie G14), the
pipeline uses **orbit-aware decomposition** via sequential conditional
sampling to correctly assign nodes to structurally distinct positions.
See the [sequential conditional sampling doc](sequential-conditional-sampling.md)
for the full algorithm.

| CCM status | Graphlets |
|------------|-----------|
| Supported (vertex-transitive) | G0, G2, G5, G8, G12, G29, C6, K6 |
| Supported (orbit-aware decomposition) | G7, G14, G17 |
| Not CCM-compatible (no Hamiltonian cycle) | G1, G3, G4, G6, G9, ... |

## The `Subgraph` class

In the new `graphs/` architecture, subgraph structures are defined via `Subgraph`
(a Pydantic model). Orbit properties are derived automatically from the subgraph's
automorphism group via igraph's VF2 algorithm.

```python
import numpy as np
from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model.sequence import SubgraphSequence

triangle = Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
print(f"Orbits: {SubgraphSequence(triangle, None).orbits}")       # [0, 0, 0]
print(f"Transitive: {SubgraphSequence(triangle, None).is_vertex_transitive}")  # True

diamond = Subgraph(adjacency=np.array([
    [0, 1, 1, 1], [1, 0, 1, 0], [1, 1, 0, 1], [1, 0, 1, 0]
]))
print(f"Orbits: {SubgraphSequence(diamond, None).orbits}")        # [0, 1, 0, 1]
print(f"Sizes: {SubgraphSequence(diamond, None).orbit_sizes}")    # {0: 2, 1: 2}
```

## Legacy `Motif` class (networks/)

The legacy `networks/` tree uses the `Motif` class with built-in graphlet constants:

```python
from craeft.networks.generation.motifs import G2, G8, G7, get_motif

# Access built-in motifs directly
triangle = G2
print(f"Triangle: {triangle.num_nodes} nodes, {triangle.num_edges} edges")

# Look up by name
k4 = get_motif("G8")
diamond = get_motif("G7")
```