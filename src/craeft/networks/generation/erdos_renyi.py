import numpy as np
from scipy.sparse import coo_matrix, csr_matrix


def random_graph(
    n: int, p: float, rng: np.random.Generator | None = None
) -> csr_matrix:
    """Generate an Erdős-Rényi random graph G(n, p).

    Args:
        n: Number of nodes.
        p: Probability of each edge existing.
        rng: Optional random generator for reproducibility.

    Returns:
        Symmetric adjacency matrix in CSR format (no self-loops).
    """
    rng = rng or np.random.default_rng()

    # Number of possible edges in upper triangle (no self-loops)
    max_edges = n * (n - 1) // 2

    # Sample how many edges, then which ones
    num_edges = rng.binomial(max_edges, p)

    if num_edges == 0:
        return csr_matrix((n, n), dtype=np.int8)

    flat_indices = rng.choice(max_edges, size=num_edges, replace=False)

    # Decode flat index to (i, j) upper-triangle coordinates
    i = ((np.sqrt(1 + 8 * flat_indices) - 1) // 2).astype(np.int64) + 1
    j = flat_indices - i * (i - 1) // 2

    # Build symmetric adjacency
    rows = np.concatenate([i, j])
    cols = np.concatenate([j, i])
    data = np.ones(2 * num_edges, dtype=np.int8)

    return coo_matrix((data, (rows, cols)), shape=(n, n), dtype=np.int8).tocsr()
