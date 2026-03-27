"""Connected component analysis for sparse adjacency matrices."""

from scipy.sparse import spmatrix
from scipy.sparse.csgraph import connected_components

MAX_CONNECTED_ATTEMPTS = 1000


class DisconnectedGraphError(RuntimeError):
    """Raised when a generator fails to produce a connected graph."""


def is_connected(adjacency: spmatrix) -> bool:
    """Check whether an undirected graph forms a single connected component.

    Args:
        adjacency: Sparse adjacency matrix (symmetric, undirected).

    Returns:
        True if the graph has exactly one connected component.
    """
    if adjacency.shape[0] <= 1:
        return True

    n_components, _ = connected_components(adjacency, directed=False)
    return n_components == 1
