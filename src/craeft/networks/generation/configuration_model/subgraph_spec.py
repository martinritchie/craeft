"""Subgraph specification for the CMA algorithm.

Pairs a motif with a per-node sequence specifying how many instances
of that motif each node participates in.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from craeft.networks.generation.motifs import Motif


@dataclass(frozen=True)
class SubgraphSpec:
    """A motif paired with per-node participation counts.

    Attributes:
        motif: The subgraph structure to embed.
        sequence: Per-node count of motif instances. Length must equal
            the number of nodes in the network.
    """

    motif: Motif
    sequence: NDArray[np.int_]

    def __post_init__(self) -> None:
        if np.any(self.sequence < 0):
            msg = "Sequence values must be non-negative"
            raise ValueError(msg)

    @property
    def total(self) -> int:
        """Total motif instances across all nodes."""
        return int(self.sequence.sum())

    @property
    def n_nodes(self) -> int:
        """Number of nodes in the network (length of sequence)."""
        return len(self.sequence)
