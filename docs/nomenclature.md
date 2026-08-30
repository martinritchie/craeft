# Nomenclature

Standard terminology used throughout the **craeft** codebase and
documentation.

## Guiding principles

1. **Consistency over legacy** — use one term throughout, even if papers used another
2. **Accessibility over jargon** — prefer terms that newcomers can understand
3. **Precision where it matters** — mathematical concepts get precise names

## Primary terms

### Graph structure

| Standard term | Definition | Avoid |
|---------------|------------|-------|
| **graph** | A collection of nodes connected by edges | network |
| **node** | A vertex in the graph | vertex (except in mathematical contexts) |
| **edge** | A connection between two nodes | link, connection, arc |
| **adjacency matrix** | Square matrix where entry (i,j) = 1 if nodes i and j are connected | |
| **degree** | Number of edges incident to a node | connectivity |
| **degree sequence** | Ordered list of all node degrees in the graph | |

"Graph" is the standard term in prose and in the current API, which exposes
`ConfigModelGraph`, `ErdosRenyiGraph` and a config → graph pipeline. The legacy
tree keeps the older name in its import paths and identifiers — `craeft.networks`,
`NetworkGenerator`, `sample_network` — and documentation quotes those verbatim.

### Subgraphs and orbits

| Standard term | Definition | Avoid |
|---------------|------------|-------|
| **subgraph** | A small, recurring pattern defined by its adjacency matrix | motif (legacy term) |
| **subgraph instance** | A specific occurrence of a subgraph pattern using particular nodes | motif instance |
| **orbit** | A set of vertices in the subgraph that are equivalent under automorphism — all vertices in an orbit have the same degree within the subgraph and receive the same structural role | corner type (legacy) |
| **orbit degree** | The degree of a vertex within the subgraph, identical for all vertices in the same orbit | cardinality (legacy) |
| **orbit size** | The number of vertices belonging to a given orbit | |
| **vertex-transitive** | All vertices belong to a single orbit (complete graphs, cycles) | uniform cardinality (legacy) |
| **participation count** | How many times a node participates in a given subgraph type | |
| **participation sequence** | Per-node participation counts for a subgraph type | |

### Algorithm components

| Standard term | Definition | Context |
|---------------|------------|---------|
| **stub** | A "half-edge" representing one unit of degree capacity | Configuration model |
| **single** | A remaining stub after subgraph allocation | CMA algorithm |
| **connector** | Stateful edge assembler that forms subgraph instances and pairs singles | CMA pipeline |
| **urn** | The shared pool of orbit slots in sequential conditional sampling | Orbit splitting |
| **allocation** | The result of verifying subgraph participations fit within degree budgets | CMA pipeline |

## Parameter naming

### Graph generation

| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| `n` | n | Number of nodes | positive integer |
| `p` | p | Edge probability (Erdos-Renyi) | [0, 1] |
| `max_retries` | — | Maximum reset-and-retry attempts when subgraph allocation fails | positive integer |

### Epidemic simulation (SIR)

| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| `tau` | $\tau$ | Transmission rate | $[0, \infty)$ |
| `gamma` | $\gamma$ | Recovery rate | $[0, \infty)$ |
| `initial_infected` | — | Number of initially infected nodes | positive integer |

## Subgraph naming

Subgraphs use [Pržulj graphlet notation](concepts/motifs.md) (G0-G29).
For subgraphs beyond 5 nodes, standard graph-theoretic notation is used
(C_n for cycles, K_n for complete graphs).

## Legacy terminology

| Legacy term | Standard term | Notes |
|-------------|---------------|-------|
| network-simulation | craeft | Original repository name |
| netsubgraph | craeft | First package name |
| motifnet | craeft | Second package name |
| network | graph | Use "graph" in prose and in new code |
| motif | subgraph | Use "subgraph" for the recurring pattern concept |
| corner / corner type | orbit | Use "orbit" for structurally equivalent vertex groups |
| cardinality | orbit degree | Use "orbit degree" |
| uniform cardinality | vertex-transitive | Use "vertex-transitive" |
| phi ($\phi$) | — | Deprecated; now controlled via participation distributions |
| TRIANGLE | G2 | Use Pržulj notation |
| TOAST / BOWTIE | G14 | Use Pržulj notation |
| K3 | G2 | Use Pržulj notation |
| K4 | G8 | Use Pržulj notation |
| SQUARE | G5 | Use Pržulj notation |
| PENTAGON | G12 | Use Pržulj notation |
| DIAMOND | G7 | Use Pržulj notation |
| HEXAGON | C6 | Extended motif (6 nodes) |