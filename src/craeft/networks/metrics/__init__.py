"""Network metrics and structural analysis."""

from craeft.networks.metrics.clustering import (
    count_triangles,
    global_clustering_coefficient,
    local_clustering,
    triangles_per_node,
)
from craeft.networks.metrics.connectivity import (
    MAX_CONNECTED_ATTEMPTS,
    DisconnectedGraphError,
    is_connected,
)

__all__ = [
    "MAX_CONNECTED_ATTEMPTS",
    "DisconnectedGraphError",
    "count_triangles",
    "global_clustering_coefficient",
    "is_connected",
    "local_clustering",
    "triangles_per_node",
]
