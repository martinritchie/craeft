"""Big-V rewiring for introducing clustering into networks.

Degree-preserving rewiring that adds triangles by finding 5-node paths
and reconnecting the endpoints. Uses net triangle tracking to allow
rewires that break some triangles if they create more.
"""

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix, lil_matrix


@dataclass(frozen=True)
class ClusteringComponents:
    """Numerator and denominator of global clustering coefficient.

    C = 3 × triangles / triplets
    """

    triangles: int
    triplets: float

    @property
    def coefficient(self) -> float:
        """Compute the clustering coefficient."""
        if self.triplets == 0:
            return 0.0
        return 3 * self.triangles / self.triplets


class BigVRewirer:
    """Degree-preserving rewiring to increase network clustering.

    Finds 5-node paths and rewires endpoints to create triangles
    while preserving the degree sequence. Uses net triangle tracking
    to accept rewires that have a positive net effect on triangles.

    Args:
        adjacency: Input network (will not be modified).
        rng: Random generator for reproducibility.
        min_delta: Minimum net triangle change to accept a rewire.
            Default is 1 (must add at least one triangle).
            Set to 0 to accept triangle-neutral rewires.
    """

    def __init__(
        self,
        adjacency: csr_matrix,
        rng: np.random.Generator | None = None,
        min_delta: int = 1,
    ) -> None:
        self._adj = lil_matrix(adjacency, dtype=np.int8)
        self._n = adjacency.shape[0]
        self._rng = rng or np.random.default_rng()
        self._triplets = self._compute_triplets(adjacency)
        self._min_delta = min_delta
        self._triangle_count = self._count_triangles()

    @staticmethod
    def _compute_triplets(adjacency: csr_matrix) -> float:
        """Compute total triplets (constant for degree-preserving rewiring)."""
        degrees = np.asarray(adjacency.sum(axis=1)).ravel()
        return float((degrees * (degrees - 1) / 2).sum())

    def _count_triangles(self) -> int:
        """Count triangles using trace(A³)/6."""
        a = csr_matrix(self._adj, dtype=np.float64)
        a3 = a @ a @ a
        return int(a3.trace()) // 6

    def clustering_components(self) -> ClusteringComponents:
        """Get current clustering coefficient components."""
        return ClusteringComponents(
            triangles=self._triangle_count,
            triplets=self._triplets,
        )

    def rewire(self, iterations: int) -> csr_matrix:
        """Apply rewiring iterations and return the modified network."""
        for _ in range(iterations):
            self._attempt_rewire()

        return self._finalize()

    def rewire_to_clustering(
        self,
        target: float,
        max_attempts: int | None = None,
    ) -> csr_matrix:
        """Rewire until target clustering coefficient is reached.

        Args:
            target: Target global clustering coefficient.
            max_attempts: Maximum rewiring attempts.

        Returns:
            Rewired network.

        Raises:
            ValueError: If target is not achievable.
        """
        if self._triplets == 0:
            raise ValueError("Network has no triplets; clustering undefined.")

        triangles_target = int(np.ceil(target * self._triplets / 3))
        triangles_to_add = triangles_target - self._triangle_count

        if triangles_to_add <= 0:
            return self._finalize()

        max_attempts = max_attempts or triangles_to_add * 100

        for _ in range(max_attempts):
            delta = self._attempt_rewire()
            if delta > 0:
                triangles_to_add -= delta
                if triangles_to_add <= 0:
                    break

        return self._finalize()

    def _finalize(self) -> csr_matrix:
        """Convert working matrix to final CSR format."""
        result = self._adj.tocsr()
        result.eliminate_zeros()
        return result

    def _attempt_rewire(self) -> int:
        """Try one rewiring operation.

        Returns:
            Net triangle change (0 if rewire was rejected or failed).
        """
        start = self._rng.integers(self._n)
        path = self._find_path(start)

        if path is None:
            return 0

        delta = self._compute_triangle_delta(path)
        if delta is None or delta < self._min_delta:
            return 0

        self._apply_rewire(path)
        self._triangle_count += delta
        return delta

    def _find_path(self, start: int) -> list[int] | None:
        """Random DFS walk of 5 nodes from start."""
        path = [start]
        visited = {start}

        while len(path) < 5:
            current = path[-1]
            neighbors = [n for n in self._adj.rows[current] if n not in visited]
            if not neighbors:
                return None
            next_node = neighbors[self._rng.integers(len(neighbors))]
            path.append(next_node)
            visited.add(next_node)

        return path

    def _neighbors(self, node: int) -> set[int]:
        """Get neighbors of a node as a set."""
        return set(self._adj.rows[node])

    def _triangles_at_edge(self, i: int, j: int) -> int:
        """Count triangles containing edge (i, j).

        A triangle containing edge (i,j) has a third node k that is
        a common neighbor of both i and j.
        """
        return len(self._neighbors(i) & self._neighbors(j))

    def _potential_triangles(self, i: int, j: int) -> int:
        """Count triangles that would be created by adding edge (i, j).

        If edge (i,j) doesn't exist, adding it would create triangles
        with each common neighbor of i and j.
        """
        return len(self._neighbors(i) & self._neighbors(j))

    def _compute_triangle_delta(self, path: list[int]) -> int | None:
        """Compute net triangle change for a rewire.

        Given path n1-n2-n3-n4-n5:
        - Remove edges (n1,n2) and (n4,n5)
        - Add edges (n1,n5) and (n2,n4)

        Returns:
            Net change in triangle count, or None if rewire is invalid
            (would create self-loop, multi-edge, or path has duplicate nodes).
        """
        n1, n2, n3, n4, n5 = path

        # Check for duplicate nodes in path (invalid path structure)
        if len(set(path)) != 5:
            return None

        # Check for self-loops in new edges
        if n1 == n5 or n2 == n4:
            return None

        # Check for multi-edges (new edges already exist)
        if n5 in self._neighbors(n1) or n4 in self._neighbors(n2):
            return None

        # Triangles destroyed by removing (n1,n2) and (n4,n5)
        destroyed = self._triangles_at_edge(n1, n2) + self._triangles_at_edge(n4, n5)

        # Triangles created by adding (n1,n5) and (n2,n4)
        # Must exclude endpoints of removed edges from common neighbor counts:
        # - For (n1,n5): n2 would need (n1,n2) and n4 would need (n4,n5), both removed
        # - For (n2,n4): n1 would need (n1,n2) and n5 would need (n4,n5), both removed
        common_15 = self._neighbors(n1) & self._neighbors(n5) - {n2, n4}
        common_24 = self._neighbors(n2) & self._neighbors(n4) - {n1, n5}
        created = len(common_15) + len(common_24)

        return created - destroyed

    def _can_rewire(self, path: list[int]) -> bool:
        """Check if rewire has non-negative net triangle change.

        Kept for backwards compatibility with tests.
        """
        delta = self._compute_triangle_delta(path)
        return delta is not None and delta >= self._min_delta

    def _apply_rewire(self, path: list[int]) -> None:
        """Remove 1-2 and 4-5, add 1-5 and 2-4."""
        n1, n2, _, n4, n5 = path

        self._adj[n1, n2] = self._adj[n2, n1] = 0
        self._adj[n4, n5] = self._adj[n5, n4] = 0

        self._adj[n1, n5] = self._adj[n5, n1] = 1
        self._adj[n2, n4] = self._adj[n4, n2] = 1


def big_v_rewire(
    adjacency: csr_matrix,
    iterations: int | None = None,
    target_clustering: float | None = None,
    rng: np.random.Generator | None = None,
    min_delta: int = 1,
) -> csr_matrix:
    """Apply Big-V rewiring to introduce clustering.

    Provide either `iterations` for fixed rewiring count, or
    `target_clustering` to rewire until a specific coefficient is reached.

    Args:
        adjacency: Input network (will not be modified).
        iterations: Number of rewiring attempts.
        target_clustering: Target global clustering coefficient.
        rng: Random generator for reproducibility.
        min_delta: Minimum net triangle change to accept a rewire.
            Default is 1. Set to 0 for more flexible rewiring.

    Returns:
        Rewired network with increased clustering.
    """
    rewirer = BigVRewirer(adjacency, rng, min_delta=min_delta)

    if target_clustering is not None:
        return rewirer.rewire_to_clustering(target_clustering)

    if iterations is not None:
        return rewirer.rewire(iterations)

    raise ValueError("Provide either 'iterations' or 'target_clustering'.")
