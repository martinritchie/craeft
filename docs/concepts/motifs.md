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

| ID | Structure | Nodes | Edges | Uniform cardinality |
|----|-----------|-------|-------|---------------------|
| **G0** | Single edge (K₂) | 2 | 1 | yes |
| **G1** | Path P₂ (wedge) | 3 | 2 | no |
| **G2** | Complete K₃ (triangle) | 3 | 3 | yes |

### 4-node graphlets

| ID | Structure | Edges | Uniform | Hamiltonian |
|----|-----------|-------|---------|-------------|
| **G3** | Path P₃ | 3 | no | no |
| **G4** | Star S₃ (claw) | 3 | no | no |
| **G5** | Cycle C₄ (square) | 4 | yes | yes |
| **G6** | Triangle + pendant (paw) | 4 | no | no |
| **G7** | K₄ − e (diamond) | 5 | no | yes |
| **G8** | Complete K₄ | 6 | yes | yes |

### 5-node graphlets (selected)

| ID | Structure | Edges | Uniform | Hamiltonian |
|----|-----------|-------|---------|-------------|
| **G12** | Cycle C₅ (pentagon) | 5 | yes | yes |
| **G14** | Two triangles sharing edge (bowtie) | 6 | no | yes |
| **G17** | Square + triangle (house) | 6 | no | yes |
| **G29** | Complete K₅ | 10 | yes | yes |

### Extended motifs (6+ nodes)

| ID | Structure | Nodes | Edges | Uniform | Hamiltonian |
|----|-----------|-------|-------|---------|-------------|
| **C6** | Cycle C₆ (hexagon) | 6 | 6 | yes | yes |
| **K6** | Complete K₆ | 6 | 15 | yes | yes |

## CCM compatibility

The CCM requires motifs with a **Hamiltonian cycle** (a cycle visiting
every node exactly once). Additionally, the current implementation only
supports **uniform cardinality** motifs — where all corners have the same
degree within the motif.

| CCM status | Graphlets |
|------------|-----------|
| Supported (uniform cardinality) | G0, G2, G5, G8, G12, G29, C6, K6 |
| Defined but blocked (mixed cardinality) | G7, G14, G17 |
| Not CCM-compatible (no Hamiltonian cycle) | G1, G3, G4, G6, G9, ... |

## The `Motif` class

Each motif is defined by its adjacency matrix and validated at construction
time. The validation checks:

1. **Square, symmetric, binary** — a valid undirected simple graph
2. **Zero diagonal** — no self-loops
3. **Hamiltonian cycle exists** — checked via backtracking DFS

Computed properties include `num_nodes`, `num_edges`, `degrees`,
`corner_types`, `cardinalities`, and `is_complete`.

## Usage

```python
from craeft import G2, G8, K6, Motif
from craeft.networks.generation.motifs import get_motif

# Access built-in motifs directly
triangle = G2
print(f"Triangle: {triangle.num_nodes} nodes, {triangle.num_edges} edges")

# Look up by name
k4 = get_motif("G8")
print(f"K4 complete: {k4.is_complete}")

# Define a custom motif
square = Motif(adjacency=[
    [0, 1, 0, 1],
    [1, 0, 1, 0],
    [0, 1, 0, 1],
    [1, 0, 1, 0],
])
```
