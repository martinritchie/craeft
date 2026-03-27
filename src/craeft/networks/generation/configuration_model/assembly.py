"""Adjacency matrix assembly from edge lists.

Step 5 of the CCM algorithm: combine motif edges and single edges
into a final sparse adjacency matrix.

Reference:
    Ritchie et al. (2014) "Higher-order structure and epidemic dynamics
    in clustered networks", Journal of Theoretical Biology 348, 21-32.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.networks.generation.configuration_model.core import edges_to_csr
from craeft.networks.generation.configuration_model.pairing import PairingResult


@dataclass(frozen=True)
class AssemblyResult:
    """Final network from CCM assembly.

    Attributes:
        adjacency: Sparse adjacency matrix (CSR format, symmetric).
        num_motif_edges: Edges contributed by motif instances.
        num_single_edges: Edges contributed by single stub pairing.
    """

    adjacency: csr_matrix
    num_motif_edges: int
    num_single_edges: int

    @property
    def num_edges(self) -> int:
        """Total edges in the network (undirected count)."""
        return int(self.adjacency.nnz // 2)


def assemble(
    num_nodes: int,
    motif_edges: tuple[NDArray[np.int_], NDArray[np.int_]],
    pairing: PairingResult,
) -> AssemblyResult:
    """Assemble adjacency matrix from motif and single edges.

    Combines edges from motif instances (Step 3) and single stub pairing
    (Step 4) into a sparse symmetric adjacency matrix. Removes any
    self-loops and collapses multi-edges.

    Args:
        num_nodes: Total nodes in the network.
        motif_edges: Tuple of (rows, cols) arrays from connector.connect().
        pairing: Result from pair_singles().

    Returns:
        AssemblyResult containing the sparse adjacency matrix and
        diagnostic edge counts.

    Example:
        >>> import numpy as np
        >>> from craeft.networks.generation.configuration_model.pairing import (
        ...     PairingResult,
        ... )
        >>> motif_edges = (np.array([0, 0, 1]), np.array([1, 2, 2]))
        >>> pairing = PairingResult(
        ...     rows=np.array([3, 4]),
        ...     cols=np.array([4, 5]),
        ...     unpaired=0,
        ... )
        >>> result = assemble(6, motif_edges, pairing)
        >>> result.num_edges
        5
    """
    motif_rows, motif_cols = motif_edges

    # Combine all edges
    all_rows = np.concatenate([motif_rows, pairing.rows])
    all_cols = np.concatenate([motif_cols, pairing.cols])

    adj = edges_to_csr(all_rows, all_cols, num_nodes)

    return AssemblyResult(
        adjacency=adj,
        num_motif_edges=len(motif_rows),
        num_single_edges=pairing.num_edges,
    )
