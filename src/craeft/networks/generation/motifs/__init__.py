"""Motif abstractions for network generation.

This module provides the core building blocks for motif-based network
generation algorithms like CMA (Cardinality Matching Algorithm) and CCM
(Clustered Configuration Model).

Key concepts:
    Motif: A subgraph defined by its adjacency matrix, with automatically
        derived properties like corner types and cardinalities.

    Built-in motifs: Pre-defined common structures using Przulj notation
        (G2 for triangle, G8 for K4, etc.) ready for use in network generation.

Naming convention follows Przulj et al. (2004) graphlet enumeration:
    G0:  Edge (2 nodes)
    G2:  Triangle / K3
    G5:  Square / 4-cycle
    G7:  Diamond (K4 minus edge)
    G8:  K4 (complete 4-graph)
    G12: Pentagon / 5-cycle
    G14: Bowtie (two triangles sharing edge)
    G17: House (square with triangle roof)
    G29: K5 (complete 5-graph)
    C6:  Hexagon / 6-cycle (extended)
    K6:  Complete 6-graph (extended)

Example:
    >>> from craeft.networks.generation.motifs import G2
    >>> G2.num_nodes
    3
    >>> G2.cardinalities
    {1: 2}
"""

from craeft.networks.generation.motifs.motif import Motif
from craeft.networks.generation.motifs.registry import (
    BUILTINS,
    C6,
    G0,
    G2,
    G5,
    G7,
    G8,
    G12,
    G14,
    G17,
    G29,
    K6,
    get_motif,
)

__all__ = [
    # Core
    "Motif",
    # Built-in motifs (Przulj notation)
    "BUILTINS",
    "C6",
    "G0",
    "G2",
    "G5",
    "G7",
    "G8",
    "G12",
    "G14",
    "G17",
    "G29",
    "K6",
    "get_motif",
]
