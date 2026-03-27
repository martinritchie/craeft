# Metrics

Network structural analysis: clustering coefficients, triangle counting,
and connectivity.

## Clustering

::: craeft.networks.metrics.clustering
    options:
      members:
        - count_triangles
        - triangles_per_node
        - local_clustering
        - global_clustering_coefficient

## Connectivity

::: craeft.networks.metrics.connectivity
    options:
      members:
        - is_connected
        - DisconnectedGraphError
        - MAX_CONNECTED_ATTEMPTS
