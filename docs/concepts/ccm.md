# Clustered Configuration Model

The Clustered Configuration Model (CCM) extends the standard configuration
model to generate networks with both a prescribed degree sequence *and* a
target clustering coefficient $\phi$.

!!! note "Reference"
    The CCM algorithm is described in: Ritchie, M., Berthouze, L., & Kiss,
    I. Z. (2014). Higher-order structure and epidemic dynamics in clustered
    networks. *Journal of Mathematical Biology*, 72(3), 483–511.

## The problem

The standard configuration model generates networks by randomly pairing
"stubs" (half-edges). This produces networks with a given degree sequence
but near-zero clustering — the probability that two neighbours of a node
are themselves connected is vanishingly small for sparse networks.

Real networks (social, biological, infrastructure) exhibit significant
clustering. The CCM addresses this by reserving a fraction of each node's
stubs for participation in small, dense substructures (motifs) before
pairing the remainder randomly.

## Algorithm pipeline

The CCM proceeds in five steps:

### Step 1: Stub allocation

Each node $i$ receives $k_i$ stubs (one per unit of degree), sampled from
the target degree distribution.

### Step 2: Partition into corners and singles

For each node, stubs are partitioned into two pools:

- **Corners** — stubs allocated to motif participation. The number of
  corners is drawn from $\text{Binomial}(\lfloor k_i / c \rfloor, \phi)$,
  where $c$ is the cardinality of the motif (the degree of each corner
  within the motif).
- **Singles** — remaining stubs, paired via the standard configuration model.

The parameter $\phi$ controls the expected fraction of stubs allocated to
motifs, and thereby the resulting clustering coefficient.

### Step 3: Form motif instances

Corner stubs are grouped and connected according to the motif's adjacency
structure. Three strategies handle collisions (when the same node appears
twice in a group):

| Strategy | Behaviour | Trade-off |
|----------|-----------|-----------|
| **Repeated** | Resample colliding groups individually | Fast, slightly biased |
| **Refuse** | Reject entire batch on any collision | Unbiased, expensive at high density |
| **Erased** | Ignore collisions, remove duplicates post-hoc | Fastest, loses edges |

### Step 4: Pair single stubs

Remaining single stubs are paired using the standard configuration model,
avoiding edges that were already created during motif formation.

### Step 5: Assemble

Motif edges and single edges are combined, symmetrised, and cleaned
(self-loops removed, multi-edges clipped) to produce the final adjacency
matrix.

## Degree-5 optimisation

For homogeneous degree-5 networks, the CCM uses an analytical allocation
that mixes K4 cliques and triangles for higher clustering accuracy:

$$\phi = \frac{2 p_3}{5}$$

where $p_3$ is the probability that a node participates in both a K4
corner and a triangle corner. This achieves higher clustering than the
general triangle-only fallback.

## Uniform cardinality constraint

The current implementation requires **uniform cardinality** — all corners
in the motif must have the same degree within the motif. This means
complete graphs (G2/triangle, G8/K4, G29/K5) and cycles (G5, G12, C6)
are supported, but mixed-cardinality motifs like G7 (diamond) and G14
(bowtie) raise `MixedCardinalityError`.

## Usage

```python
import numpy as np
from craeft import configuration_model, global_clustering_coefficient

rng = np.random.default_rng(42)
degrees = np.full(1000, 5)

# phi controls clustering: 0.0 = unclustered, higher = more clustered
for phi in [0.0, 0.1, 0.2, 0.3]:
    adj = configuration_model(degrees, phi=phi, rng=rng)
    cc = global_clustering_coefficient(adj)
    print(f"phi={phi:.1f} -> clustering={cc:.3f}")
```
