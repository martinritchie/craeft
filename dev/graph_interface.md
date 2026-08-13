# Configuration Model — Implementation Progress

## Package structure

```
graphs/
├── base.py                                    ✅ implemented + tested
├── erdos_renyi.py                             ✅ implemented + tested
├── metrics/
│   ├── __init__.py                            ✅ implemented
│   └── clustering.py                          ✅ implemented + tested
└── configuration_model/
    ├── __init__.py                            ✅ implemented
    ├── models.py                              ✅ implemented + tested
    ├── connection.py                          ✅ implemented + tested
    └── sequence/
        ├── __init__.py                        ✅ implemented
        ├── sampling.py                        ✅ implemented + tested
        ├── subgraph_sequence.py               ✅ implemented + tested
        └── allocation.py                      ✅ implemented + tested
```

All stubs are filled. All tests written. `make docs` builds clean.

## Module status

### models.py — ConfigModelConfig + ConfigModelGraph

| Component | Status | Notes |
|---|---|---|
| `ConfigModelConfig` validation | ✅ | degrees length, non-negative, even sum |
| `ConfigModelGraph.from_config` (no sequences) | ✅ | delegates to Connector |
| `ConfigModelGraph.from_config` (with sequences) | ✅ | full CMA pipeline: sample → split → allocate → connect subgraphs → connect singles → assemble |
| `ConfigModelGraph.clustering_coefficient` | ✅ | delegates to global_clustering_coefficient |

### connection.py — Connector

| Component | Status | Notes |
|---|---|---|
| `Connector.__init__` | ✅ | tracks n, rng, edges, existing set |
| `Connector.connect_singles` | ✅ | stub pairing with self-loop/multi-edge/existing-edge filtering |
| `Connector.connect_subgraph` | ✅ | per-orbit pools, batch-rejection on collision (duplicates + existing edges), commit from originals |
| `Connector.to_csr` | ✅ | symmetric COO → CSR assembly |

### sequence/sampling.py

| Component | Status | Notes |
|---|---|---|
| `sample_degree_sequence` | ✅ tested | rejection sampling, even sum, bounded |
| `_sample_sequence` | ✅ tested | shared impl with divisor + max_iterations safety valve (10000) |

### sequence/subgraph_sequence.py — SubgraphSequence

| Component | Status | Notes |
|---|---|---|
| `SubgraphSequence.sample` | ✅ | delegates to _sample_sequence, max_value=n for participation counts |
| `SubgraphSequence._split_by_orbit` | ✅ tested | sequential conditional sampling: k=1 no-op, k=2 closed-form [ℓ,u], k≥3 lattice point enumeration |
| `SubgraphSequence.orbits` | ✅ tested | VF2 automorphism via igraph, cached in __post_init__ |
| `SubgraphSequence.orbit_degrees` | ✅ tested | cached, derived from orbits + subgraph degrees |
| `SubgraphSequence.orbit_sizes` | ✅ tested | cached, count per orbit |
| `SubgraphSequence.is_vertex_transitive` | ✅ tested | cached, single-orbit check |
| `SubgraphSequence.edges_for` | ✅ tested | triu_indices mask → concrete node ID mapping |

### sequence/allocation.py — Allocation + allocate_subgraphs

| Component | Status | Notes |
|---|---|---|
| `Allocation.node_ids_for` | ✅ | np.repeat expansion |
| `Allocation.single_stubs` | ✅ | np.repeat expansion |
| `allocate_subgraphs` | ✅ tested | degree budget verification, builds bins, computes singles |
| `AllocationError` | ✅ | exception class defined |

## Tests

| Area | Status | Count |
|---|---|---|
| Base classes (BaseGraph, Subgraph, etc.) | ✅ | ~70 |
| Clustering metrics | ✅ | ~26 |
| Erdos-Renyi | ✅ | ~21 |
| Degree sequence sampling | ✅ | ~46 |
| Connector.connect_singles | ✅ | ~21 |
| ConfigModelGraph (no sequences) | ✅ | ~16 |
| Orbit computation (orbits, degrees, sizes, transitivity) | ✅ | 39 |
| edges_for | ✅ | 4 |
| _split_by_orbit (vertex-transitive + diamond) | ✅ | 9 |
| ConfigModelGraph (with sequences — triangle, diamond) | ✅ | 5 |
| **Total** | **✅** | **574** |

## Implementation order (completed)

1. ✅ Orbit computation (VF2 + union-find, cached in __post_init__)
2. ✅ edges_for (triu_indices + mask)
3. ✅ Allocation.node_ids_for + single_stubs (simple helpers)
4. ✅ _split_by_orbit (sequential urn sampling: k=1, k=2 closed-form, k≥3 lattice points)
5. ✅ allocate_subgraphs (degree budget verification)
6. ✅ Connector.connect_subgraph (per-orbit pools, batch-rejection, commit-from-originals)
7. ✅ ConfigModelGraph.from_config sequences path (full pipeline + retry loop)

## Key implementation details

- **Orbit caching**: VF2 automorphism enumeration is expensive. Computed once in
  `__post_init__` and cached for all orbit-related properties.
- **Sequential conditional sampling**: Non-transitive subgraphs (diamond, bowtie)
  use an urn-based algorithm. The urn starts with M·σ_o slots per orbit; nodes
  draw in descending participation order. For k=2 orbits, a closed-form bound
  [ℓ_i, u_i] gives uniform sampling in O(n). For k≥3, lattice points of the
  local polytope are enumerated. Design doc at `docs/concepts/sequential-conditional-sampling.md`.
- **Batch-rejection connection**: `connect_subgraph` validates all instances on
  working copies before committing from the originals. This avoids the partial-
  commit bug where collision recovery left pools in an inconsistent state.
- **Safety valve**: `_sample_sequence` has `max_iterations=10000` to prevent
  infinite loops when the distribution is incompatible with bounds (caught as
  `RuntimeError` and retried in the outer loop).
- **Retry architecture**: `from_config` catches `AllocationError`, `ConnectionError`,
  `ValueError`, and `RuntimeError`, resampling participation sequences up to
  `max_retries` times.
- **Dual-tree coexistence**: New `graphs/` tree (config→graph pipeline) lives
  alongside legacy `networks/` tree (function-based API + generator dataclasses).
  Docs reference both.

## Design decisions (unchanged from original plan)

- One graph class (`ConfigModelGraph`), not two — sequences=() gives standard CM
- `SubgraphSequence` holds distribution, not raw array — sampling at generation time
- Orbits computed from automorphism group, not degree approximation
- `Connector` is stateful — accumulates edges, tracks existing set across steps
- Failure mode: raise + retry, not silent shedding
- Standard graph theory vocabulary (orbits, vertex-transitive), not paper jargon

## References

- Ritchie et al. (2015) "Generation and analysis of networks with a prescribed
  degree sequence and subgraph family", J. Complex Networks 5(1), 1-31.
- Ritchie et al. (2016) "Beyond clustering: mean-field dynamics on networks with
  arbitrary subgraph composition", J. Math. Biol. 72, 255-281.