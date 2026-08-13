# Metrics

Network structural analysis: clustering coefficients and connectivity.

## Clustering (new API)

::: craeft.graphs.metrics.clustering
    options:
      members:
        - global_clustering_coefficient

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