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
