# Nomenclature

Standard terminology used throughout the **craeft** codebase and
documentation.

## Guiding principles

1. **Consistency over legacy** — use one term throughout, even if papers used another
2. **Accessibility over jargon** — prefer terms that newcomers can understand
3. **Precision where it matters** — mathematical concepts get precise names

## Primary terms

### Network structure

| Standard term | Definition | Avoid |
|---------------|------------|-------|
| **network** | A collection of nodes connected by edges | graph (except in mathematical definitions) |
| **node** | A vertex in the network | vertex (except in mathematical contexts) |
| **edge** | A connection between two nodes | link, connection, arc |
| **adjacency matrix** | Square matrix where entry (i,j) = 1 if nodes i and j are connected | |
| **degree** | Number of edges incident to a node | connectivity |
| **degree sequence** | Ordered list of all node degrees in the network | |

### Motifs

| Standard term | Definition | Avoid |
|---------------|------------|-------|
| **motif** | A small, recurring substructure pattern in a network | subgraph (except for the mathematical concept) |
| **motif instance** | A specific occurrence of a motif pattern using particular nodes | subgraph instance |
| **corner** | A node's participation slot in a motif instance | position, vertex |
| **corner type** | Classification of corners by their degree within the motif | |
| **cardinality** | The degree of a corner within its motif (edges contributed) | |

### Algorithm components

| Standard term | Definition | Context |
|---------------|------------|---------|
| **stub** | A "half-edge" representing one unit of degree capacity | Configuration model |
| **hyperstub** | A stub allocated to participate in motif formation | CCM algorithm |
| **single** | A remaining stub after motif allocation | CCM algorithm |
| **connector** | Strategy for forming motif instances from hyperstubs | CCM Step 3 |
| **pairer** | Strategy for pairing single stubs into edges | CCM Step 4 |

## Parameter naming

### Network generation

| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| `n` | n | Number of nodes | positive integer |
| `phi` | $\phi$ | Target clustering coefficient | [0, 1] |
| `p` | p | Edge probability (Erdos-Renyi) | [0, 1] |

### Epidemic simulation (SIR)

| Parameter | Symbol | Description | Range |
|-----------|--------|-------------|-------|
| `tau` | $\tau$ | Transmission rate | $[0, \infty)$ |
| `gamma` | $\gamma$ | Recovery rate | $[0, \infty)$ |
| `initial_infected` | — | Number of initially infected nodes | positive integer |

## Motif naming

Motifs use [Pržulj graphlet notation](concepts/motifs.md) (G0-G29).
For motifs beyond 5 nodes, standard graph-theoretic notation is used
(C_n for cycles, K_n for complete graphs).

## Legacy terminology

| Legacy term | Standard term | Notes |
|-------------|---------------|-------|
| network-simulation | craeft | Original repository name |
| netsubgraph | craeft | First package name |
| motifnet | craeft | Second package name |
| graph | network | Use "network" in code |
| subgraph | motif | Use "motif" for the recurring pattern concept |
| TRIANGLE | G2 | Use Pržulj notation |
| TOAST / BOWTIE | G14 | Use Pržulj notation |
| K3 | G2 | Use Pržulj notation |
| K4 | G8 | Use Pržulj notation |
| SQUARE | G5 | Use Pržulj notation |
| PENTAGON | G12 | Use Pržulj notation |
| DIAMOND | G7 | Use Pržulj notation |
| HEXAGON | C6 | Extended motif (6 nodes) |
