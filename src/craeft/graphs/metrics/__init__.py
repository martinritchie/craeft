"""Graph metrics and structural analysis."""

from craeft.graphs.metrics.clustering import (
    count_triangles,
    global_clustering_coefficient,
    local_clustering,
    triangles_per_node,
)
from craeft.graphs.metrics.correlation import (
    average_neighbour_degree,
    clustering_by_degree,
    degree_assortativity,
)
from craeft.graphs.metrics.cycles import (
    cycles_per_node,
    induced_cycle_count,
)
from craeft.graphs.metrics.order_four import (
    count_order_four,
    order_four_per_node,
    order_four_ratios,
)
from craeft.graphs.metrics.subgraph import (
    designed_clustering,
    designed_triangles,
    triangles_per_orbit,
    unique_triangles,
)

__all__ = [
    "average_neighbour_degree",
    "clustering_by_degree",
    "count_order_four",
    "count_triangles",
    "cycles_per_node",
    "degree_assortativity",
    "designed_clustering",
    "designed_triangles",
    "global_clustering_coefficient",
    "induced_cycle_count",
    "local_clustering",
    "order_four_per_node",
    "order_four_ratios",
    "triangles_per_node",
    "triangles_per_orbit",
    "unique_triangles",
]
