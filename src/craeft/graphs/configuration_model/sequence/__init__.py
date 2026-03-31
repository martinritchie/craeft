"""Sequence sampling, orbit splitting, and allocation.

Modules:
    sampling.py            Rejection sampling for degree and participation sequences.
    subgraph_sequence.py   SubgraphSequence type with orbit properties.
    allocation.py          Greedy subgraph allocation to nodes.
"""

from craeft.graphs.configuration_model.sequence.allocation import (
    Allocation,
    AllocationError,
    allocate_subgraphs,
)
from craeft.graphs.configuration_model.sequence.sampling import (
    sample_degree_sequence,
)
from craeft.graphs.configuration_model.sequence.subgraph_sequence import (
    SubgraphSequence,
)

__all__ = [
    "Allocation",
    "AllocationError",
    "SubgraphSequence",
    "allocate_subgraphs",
    "sample_degree_sequence",
]
