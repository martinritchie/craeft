"""Pre-defined motifs for network generation using Przulj notation.

These motifs are validated on import to ensure they satisfy the
Hamiltonian cycle requirement for CMA compatibility.

Naming convention follows Przulj et al. (2004) graphlet enumeration:
- G0-G29: Graphlets on 2-5 nodes
- C6: 6-cycle (extension)
- K6: Complete graph on 6 nodes (extension)

Reference:
    Przulj, N., Corneil, D. G., & Jurisica, I. (2004).
    Modeling interactome: scale-free or geometric?
    Bioinformatics, 20(18), 3508-3515.
"""

import numpy as np

from craeft.networks.generation.motifs.motif import Motif

# -----------------------------------------------------------------------------
# Complete graphs (all corners equivalent)
# -----------------------------------------------------------------------------

G0 = Motif(
    adjacency=np.array(
        [
            [0, 1],
            [1, 0],
        ]
    )
)
"""G0: Simple edge (2-node graphlet). Cardinality: 1."""

G2 = Motif(
    adjacency=np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ]
    )
)
"""G2: Triangle / K3 (3-node complete graph). All corners have cardinality 2."""

G8 = Motif(
    adjacency=np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
    )
)
"""G8: Complete graph K4 (4-node). All corners have cardinality 3."""

G29 = Motif(
    adjacency=np.array(
        [
            [0, 1, 1, 1, 1],
            [1, 0, 1, 1, 1],
            [1, 1, 0, 1, 1],
            [1, 1, 1, 0, 1],
            [1, 1, 1, 1, 0],
        ]
    )
)
"""G29: Complete graph K5 (5-node). All corners have cardinality 4."""

K6 = Motif(
    adjacency=np.array(
        [
            [0, 1, 1, 1, 1, 1],
            [1, 0, 1, 1, 1, 1],
            [1, 1, 0, 1, 1, 1],
            [1, 1, 1, 0, 1, 1],
            [1, 1, 1, 1, 0, 1],
            [1, 1, 1, 1, 1, 0],
        ]
    )
)
"""K6: Complete graph on 6 nodes (extended notation). All corners have cardinality 5."""


# -----------------------------------------------------------------------------
# Cycles (Hamiltonian by definition)
# -----------------------------------------------------------------------------

G5 = Motif(
    adjacency=np.array(
        [
            [0, 1, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [1, 0, 1, 0],
        ]
    )
)
"""G5: 4-cycle (square). All corners have cardinality 2."""

G12 = Motif(
    adjacency=np.array(
        [
            [0, 1, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 1],
            [1, 0, 0, 1, 0],
        ]
    )
)
"""G12: 5-cycle (pentagon). All corners have cardinality 2."""

C6 = Motif(
    adjacency=np.array(
        [
            [0, 1, 0, 0, 0, 1],
            [1, 0, 1, 0, 0, 0],
            [0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 1, 0],
            [0, 0, 0, 1, 0, 1],
            [1, 0, 0, 0, 1, 0],
        ]
    )
)
"""C6: 6-cycle (hexagon, extended notation). All corners have cardinality 2."""


# -----------------------------------------------------------------------------
# Incomplete subgraphs (multiple corner types)
# -----------------------------------------------------------------------------

# G14 structure (two triangles sharing an edge):
#
#     0 --- 1 --- 3
#     |     |     |
#     +--2--+--4--+
#
# Hamiltonian cycle: 0 -> 1 -> 3 -> 4 -> 2 -> 0
# Degrees: [2, 3, 3, 2, 2]
# Corner types: type 1 (degree 2) for nodes 0, 3, 4
#               type 2 (degree 3) for nodes 1, 2

G14 = Motif(
    adjacency=np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ]
    )
)
"""G14: Two triangles sharing an edge (bowtie/toast). Has two corner types."""


# G17 structure (square with a triangle on top):
#
#       0
#      / \
#     1---2
#     |   |
#     3---4
#
# Hamiltonian cycle: 0 -> 1 -> 3 -> 4 -> 2 -> 0
# Degrees: [2, 3, 3, 2, 2]

G17 = Motif(
    adjacency=np.array(
        [
            [0, 1, 1, 0, 0],
            [1, 0, 1, 1, 0],
            [1, 1, 0, 0, 1],
            [0, 1, 0, 0, 1],
            [0, 0, 1, 1, 0],
        ]
    )
)
"""G17: House (square with triangle roof). Isomorphic to G14."""


# G7 (K4 minus one edge):
#
#     0
#    /|\
#   1-+-2
#    \|/
#     3
#
# Missing edge: 1-2
# Hamiltonian cycle: 0 -> 1 -> 3 -> 2 -> 0
# Degrees: [3, 2, 2, 3]

G7 = Motif(
    adjacency=np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 0],
        ]
    )
)
# G7: Diamond (K4 minus one edge).
# Two corner types: hubs (degree 3) and leaves (degree 2).


# -----------------------------------------------------------------------------
# Registry for lookup by name
# -----------------------------------------------------------------------------

BUILTINS: dict[str, Motif] = {
    "g0": G0,
    "g2": G2,
    "g5": G5,
    "g7": G7,
    "g8": G8,
    "g12": G12,
    "g14": G14,
    "g17": G17,
    "g29": G29,
    "c6": C6,
    "k6": K6,
}


def get_motif(name: str) -> Motif:
    """Get a built-in motif by name.

    Args:
        name: Case-insensitive motif name using Przulj notation
            (G0, G2, G5, G7, G8, G12, G14, G17, G29, C6, K6).

    Returns:
        The corresponding Motif object.

    Raises:
        KeyError: If no motif with that name exists.

    Example:
        >>> triangle = get_motif("G2")
        >>> triangle.num_nodes
        3
    """
    key = name.lower()
    if key not in BUILTINS:
        available = ", ".join(sorted(BUILTINS.keys()))
        msg = f"Unknown motif '{name}'. Available: {available}"
        raise KeyError(msg)
    return BUILTINS[key]
