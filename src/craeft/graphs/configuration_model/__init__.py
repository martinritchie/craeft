"""Configuration model for generating random graphs with prescribed degree sequences.

Algorithm:
    Degree sequence → [split by orbit → allocate to nodes →
    connect subgraph instances →] pair remaining stubs →
    adjacency matrix.

    The bracketed steps apply when subgraph sequences are specified.
    Without them, this reduces to the standard configuration model.

Modules:
    models.py      Config and graph class (user-facing types).
    sequence.py    Sequence sampling, orbit splitting, and allocation.
    connection.py  Edge formation from subgraph allocations and singles.
"""

from craeft.graphs.configuration_model.models import (
    ConfigModelConfig,
    ConfigModelGraph,
)
from craeft.graphs.configuration_model.sequence import (
    SubgraphSequence,
)

__all__ = [
    "ConfigModelConfig",
    "ConfigModelGraph",
    "SubgraphSequence",
]
