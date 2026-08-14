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
