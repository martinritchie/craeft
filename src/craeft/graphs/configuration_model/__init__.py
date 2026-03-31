"""Configuration model for generating random graphs with prescribed degree sequences.

Two variants:

    Vanilla CM (ConfigModelGraph):
        Degree sequence → stub pairing → adjacency matrix.

    CMA (CMAGraph):
        Degree sequence + subgraph sequences → split by orbit
        → greedy allocation to nodes → connect subgraph instances
        → pair remaining singles → adjacency matrix.

Modules:
    models.py      Configs and graph classes (user-facing types).
    sequence.py    Sequence sampling, orbit splitting, and matching.
    connection.py  Edge formation from subgraph allocations and singles.
"""

from craeft.graphs.configuration_model.models import (
    CMAConfig,
    CMAGraph,
    ConfigModelConfig,
    ConfigModelGraph,
)
from craeft.graphs.configuration_model.sequence import (
    SubgraphSequence,
)

__all__ = [
    "CMAConfig",
    "CMAGraph",
    "ConfigModelConfig",
    "ConfigModelGraph",
    "SubgraphSequence",
]
