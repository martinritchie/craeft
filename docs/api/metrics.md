# Metrics

Network structural analysis: clustering, degree correlation, order-four structure,
and connectivity.

## Clustering (new API)

::: craeft.graphs.metrics.clustering
    options:
      members:
        - count_triangles
        - triangles_per_node
        - local_clustering
        - global_clustering_coefficient

## Designed clustering

Closed-form clustering metrics computed from a `ConfigModelConfig`, before any
graph is generated. See `docs/concepts/ccm.md` for a worked example.

::: craeft.graphs.metrics.subgraph
    options:
      members:
        - unique_triangles
        - triangles_per_orbit
        - designed_triangles
        - designed_clustering

## Degree correlation

::: craeft.graphs.metrics.correlation
    options:
      members:
        - degree_assortativity
        - average_neighbour_degree
        - clustering_by_degree

## Order-four structure

Ratios and per-node counts of connected 4-node induced subgraphs (2014
Section 2.2, item 4). Clustering is the *matched* quantity in a dataset
pair; order-four composition is what actually distinguishes networks with
identical degree distribution and identical clustering.

### Numbering

`order_four_ratios` follows the published 2014 numbering:

| Key | Meaning |
|:--|:--|
| `phi_4_1` | **all closed quadruples** (the aggregate) |
| `phi_4_2` | empty square (chordless 4-cycle) |
| `phi_4_3` | square with one diagonal (diamond) |
| `phi_4_4` | complete square (K₄) |
| `unclosed` | `1 - phi_4_1` — path, star and paw combined |
| `paw` | triangle + pendant edge; a *component* of `unclosed`, not a further term |

with `phi_4_1 == phi_4_2 + phi_4_3 + phi_4_4`.

!!! warning "Breaking change (ticket 008)"

    This module previously returned the **empty square** under the key
    `phi_4_1` and returned no aggregate at all — a transcription error, not
    the paper's numbering. Any recorded `phi_4_1` value from before that fix
    should be read as `phi_4_2`.

### Denominator, and comparability with 2014 Table 2

The denominator is the count of connected 4-node **induced subgraphs**, all
six isomorphism classes, stars included. The 2014 paper gives two different
denominators one sentence apart, and its Appendix A.2 algorithm implements
the other one — a count of 4-node *paths*, which a star has none of. Stars
are 21–27% of craeft's denominator, so **these values are internally
comparable across generated families but are not numerically comparable with
2014 Table 2.** A paper-faithful path-count variant is deliberately not
implemented; see the module docstring for the full argument.

The paw is catalogued as **G6** in
[Motifs](../concepts/motifs.md) — non-Hamiltonian, and therefore structurally
impossible as an *input* subgraph to the CCM. Every paw observed in a
generated graph is a by-product, which is what makes it a sharp diagnostic
for triangle-bearing families.

::: craeft.graphs.metrics.order_four
    options:
      members:
        - count_order_four
        - order_four_ratios
        - order_four_per_node

## Clustering (legacy)

::: craeft.networks.metrics.clustering
    options:
      members:
        - count_triangles
        - triangles_per_node
        - local_clustering
        - global_clustering_coefficient

## Connectivity (legacy)

::: craeft.networks.metrics.connectivity
    options:
      members:
        - is_connected
        - DisconnectedGraphError
        - MAX_CONNECTED_ATTEMPTS
