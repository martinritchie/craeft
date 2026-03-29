"""Configuration model for generating random graphs with prescribed degree sequences."""

from craeft.graphs.configuration_model.graph import (
    ConfigModelConfig,
    ConfigModelGraph,
)
from craeft.graphs.configuration_model.pairing import (
    pair_stubs,
)
from craeft.graphs.configuration_model.sequence import (
    sample_degree_sequence,
)

__all__ = [
    "ConfigModelConfig",
    "ConfigModelGraph",
    "pair_stubs",
    "sample_degree_sequence",
]
