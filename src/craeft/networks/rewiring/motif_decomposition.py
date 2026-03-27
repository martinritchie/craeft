"""Motif Decomposition algorithm for generating networks with target clustering.

Implements the "tearing" algorithm from Ritchie et al. (2014) Section 2.1.2.
Starts with disconnected cliques (clustering = 1.0) and iteratively rewires
edges to reduce clustering to a target level.

Uses incremental triangle tracking for efficient clustering coefficient updates.
"""

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


def motif_decomposition(
    num_nodes: int,
    clique_size: int,
    target_clustering: float,
    *,
    rng: np.random.Generator | None = None,
    max_iterations: int = 100_000,
) -> csr_matrix:
    """Generate network via motif decomposition.

    Starts with disconnected complete subgraphs (cliques) and iteratively
    rewires edges to reduce clustering from 1.0 to the target level.

    Args:
        num_nodes: Total nodes N (must be divisible by clique_size).
        clique_size: Size of initial cliques m (e.g., 3 for triangles, 4 for K4).
        target_clustering: Desired global clustering coefficient (0 to 1).
        rng: Random generator for reproducibility.
        max_iterations: Safety limit to prevent infinite loops.

    Returns:
        Sparse adjacency matrix with target clustering.

    Raises:
        ValueError: If num_nodes is not divisible by clique_size.
        ValueError: If target_clustering is not in [0, 1].
        ValueError: If clique_size < 2.
    """
    _validate_inputs(num_nodes, clique_size, target_clustering)
    rng = rng or np.random.default_rng()

    # Initialize disconnected cliques
    adj, local_edges = _create_cliques(num_nodes, clique_size)
    global_edges: set[tuple[int, int]] = set()

    # Initialize triangle and wedge counts for incremental tracking
    # Wedge count stays constant since edge swaps preserve degree
    triangle_count = _initial_triangle_count(num_nodes, clique_size)
    wedge_count = _compute_wedge_count(adj)

    # Handle edge case: no wedges means clustering is undefined
    if wedge_count == 0:
        return adj.tocsr()

    # Check if target is already achieved
    current_clustering = 3 * triangle_count / wedge_count
    if current_clustering <= target_clustering:
        return adj.tocsr()

    # First swap: two local edges → two global edges
    if len(local_edges) < 2:
        return adj.tocsr()

    e1, e2 = _sample_two_edges(local_edges, rng)

    # Count triangles before swap
    old_triangles = _count_triangles_at_edges(adj, e1, e2)
    result = _swap_edges(adj, e1, e2, rng)

    if result is not None:
        new_e1, new_e2 = result
        # Count triangles after swap and update
        new_triangles = _count_triangles_at_edges(adj, new_e1, new_e2)
        triangle_count += new_triangles - old_triangles

        local_edges.discard(e1)
        local_edges.discard(e2)
        global_edges.add(new_e1)
        global_edges.add(new_e2)

    # Iterate: one local + one global → two global
    for _ in range(max_iterations):
        current_clustering = 3 * triangle_count / wedge_count
        if current_clustering <= target_clustering:
            break

        if not local_edges or not global_edges:
            break

        local_e = _sample_one_edge(local_edges, rng)
        global_e = _sample_one_edge(global_edges, rng)

        # Count triangles before swap
        old_triangles = _count_triangles_at_edges(adj, local_e, global_e)
        result = _swap_edges(adj, local_e, global_e, rng)

        if result is None:
            continue

        new_e1, new_e2 = result
        # Count triangles after swap and update
        new_triangles = _count_triangles_at_edges(adj, new_e1, new_e2)
        triangle_count += new_triangles - old_triangles

        local_edges.discard(local_e)
        global_edges.discard(global_e)
        global_edges.add(new_e1)
        global_edges.add(new_e2)

    return adj.tocsr()


def _validate_inputs(
    num_nodes: int, clique_size: int, target_clustering: float
) -> None:
    """Validate algorithm inputs."""
    if clique_size < 2:
        msg = f"clique_size must be >= 2, got {clique_size}"
        raise ValueError(msg)

    if num_nodes % clique_size != 0:
        msg = (
            f"num_nodes ({num_nodes}) must be divisible by clique_size ({clique_size})"
        )
        raise ValueError(msg)

    if not 0 <= target_clustering <= 1:
        msg = f"target_clustering must be in [0, 1], got {target_clustering}"
        raise ValueError(msg)


def _initial_triangle_count(num_nodes: int, clique_size: int) -> int:
    """Compute triangle count for disconnected cliques.

    Each clique of size m contains C(m, 3) = m(m-1)(m-2)/6 triangles.
    """
    num_cliques = num_nodes // clique_size
    triangles_per_clique = clique_size * (clique_size - 1) * (clique_size - 2) // 6
    return num_cliques * triangles_per_clique


def _compute_wedge_count(adj: lil_matrix) -> int:
    """Compute total number of wedges (paths of length 2).

    A wedge centered at node v is a pair of edges sharing v.
    Count = sum over all nodes of C(degree, 2).

    This value stays constant during edge swaps since degree is preserved.
    """
    total = 0
    for i in range(adj.shape[0]):
        degree = len(adj.rows[i])
        total += degree * (degree - 1) // 2
    return total


def _count_triangles_at_edges(
    adj: lil_matrix, e1: tuple[int, int], e2: tuple[int, int]
) -> int:
    """Count triangles containing at least one of the two edges.

    A triangle containing edge (u, v) has a third node w that is
    a common neighbor of u and v.

    Handles the case where edges share a node to avoid double-counting.
    """
    a, b = e1
    c, d = e2

    def neighbors(node: int) -> set[int]:
        return set(adj.rows[node])

    # Triangles containing edge (a,b): common neighbors of a and b
    neighbors_a = neighbors(a)
    neighbors_b = neighbors(b)
    common_ab = neighbors_a & neighbors_b

    # Triangles containing edge (c,d): common neighbors of c and d
    neighbors_c = neighbors(c)
    neighbors_d = neighbors(d)
    common_cd = neighbors_c & neighbors_d

    total = len(common_ab) + len(common_cd)

    # Correct for double-counting if edges share exactly one node
    # If edges are (a,b) and (b,c), triangle (a,b,c) is counted in both terms
    shared_nodes = {a, b} & {c, d}
    if len(shared_nodes) == 1:
        other_from_e1 = ({a, b} - shared_nodes).pop()
        other_from_e2 = ({c, d} - shared_nodes).pop()

        # Check if these non-shared endpoints form an edge (completing a triangle)
        if other_from_e2 in neighbors(other_from_e1):
            total -= 1

    return total


def _create_cliques(
    num_nodes: int, clique_size: int
) -> tuple[lil_matrix, set[tuple[int, int]]]:
    """Initialize disconnected complete subgraphs (cliques).

    Args:
        num_nodes: Total number of nodes.
        clique_size: Size of each clique.

    Returns:
        Tuple of (adjacency matrix, set of edges as canonical tuples).
    """
    adj = lil_matrix((num_nodes, num_nodes), dtype=np.int8)
    edges: set[tuple[int, int]] = set()

    num_cliques = num_nodes // clique_size

    for clique_idx in range(num_cliques):
        start = clique_idx * clique_size
        nodes = list(range(start, start + clique_size))

        # Connect all pairs within the clique
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                adj[u, v] = 1
                adj[v, u] = 1
                edges.add(_canonical_edge(u, v))

    return adj, edges


def _canonical_edge(i: int, j: int) -> tuple[int, int]:
    """Return edge as (min, max) for consistent set operations."""
    return (min(i, j), max(i, j))


def _sample_one_edge(
    edges: set[tuple[int, int]], rng: np.random.Generator
) -> tuple[int, int]:
    """Sample one edge uniformly from the set."""
    edge_list = list(edges)
    idx = rng.integers(len(edge_list))
    return edge_list[idx]


def _sample_two_edges(
    edges: set[tuple[int, int]], rng: np.random.Generator
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Sample two distinct edges uniformly from the set."""
    edge_list = list(edges)
    indices = rng.choice(len(edge_list), size=2, replace=False)
    return edge_list[indices[0]], edge_list[indices[1]]


def _swap_edges(
    adj: lil_matrix,
    e1: tuple[int, int],
    e2: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    """Swap endpoints of two edges.

    Given edges (a, b) and (c, d), creates either:
    - (a, c) and (b, d), OR
    - (a, d) and (b, c)

    The choice is made randomly. Returns None if the swap would create
    a self-loop or multi-edge.

    Args:
        adj: Adjacency matrix (will be modified in place).
        e1: First edge as (node, node) tuple.
        e2: Second edge as (node, node) tuple.
        rng: Random generator.

    Returns:
        Tuple of two new edges if successful, None if swap is invalid.
    """
    a, b = e1
    c, d = e2

    # Randomly choose swap configuration
    if rng.random() < 0.5:
        new_e1 = (a, c)
        new_e2 = (b, d)
    else:
        new_e1 = (a, d)
        new_e2 = (b, c)

    # Check for self-loops
    if new_e1[0] == new_e1[1] or new_e2[0] == new_e2[1]:
        return None

    # Canonicalize for checking
    new_e1 = _canonical_edge(*new_e1)
    new_e2 = _canonical_edge(*new_e2)

    # Check for multi-edges (edge already exists)
    if adj[new_e1[0], new_e1[1]] != 0 or adj[new_e2[0], new_e2[1]] != 0:
        return None

    # Check if new edges are the same as each other
    if new_e1 == new_e2:
        return None

    # Perform the swap
    adj[a, b] = adj[b, a] = 0
    adj[c, d] = adj[d, c] = 0
    adj[new_e1[0], new_e1[1]] = adj[new_e1[1], new_e1[0]] = 1
    adj[new_e2[0], new_e2[1]] = adj[new_e2[1], new_e2[0]] = 1

    return new_e1, new_e2
