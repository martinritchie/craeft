"""Motif: A subgraph defined by its adjacency matrix.

All properties (corner types, cardinalities) are derived automatically.
Validation ensures the motif has a Hamiltonian cycle for CMA compatibility.
"""

from typing import Annotated, Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, BeforeValidator, computed_field
from pydantic.functional_validators import AfterValidator


def _to_numpy(v: Any) -> NDArray[np.int_]:
    """Convert input to numpy array."""
    return np.asarray(v, dtype=np.int_)


def _has_hamiltonian_cycle(adj: NDArray[np.int_]) -> bool:
    """Check if graph has a Hamiltonian cycle using backtracking.

    A Hamiltonian cycle visits every node exactly once and returns to start.
    This property ensures sufficient connectivity for the CMA algorithm.
    """
    n = adj.shape[0]
    if n < 3:
        return n == 2 and bool(adj[0, 1] == 1)

    def backtrack(path: list[int], visited: set[int]) -> bool:
        if len(path) == n:
            return bool(adj[path[-1], path[0]] == 1)

        current = path[-1]
        for next_node in range(n):
            if next_node not in visited and adj[current, next_node] == 1:
                path.append(next_node)
                visited.add(next_node)
                if backtrack(path, visited):
                    return True
                path.pop()
                visited.remove(next_node)
        return False

    return backtrack([0], {0})


def _validate_adjacency(adj: NDArray[np.int_]) -> NDArray[np.int_]:
    """Validate adjacency matrix structure and Hamiltonian property."""
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        msg = f"Adjacency must be square, got shape {adj.shape}"
        raise ValueError(msg)

    if not np.allclose(adj, adj.T):
        msg = "Adjacency matrix must be symmetric"
        raise ValueError(msg)

    if not np.all((adj == 0) | (adj == 1)):
        msg = "Adjacency matrix must be binary (0s and 1s)"
        raise ValueError(msg)

    if np.any(np.diag(adj) != 0):
        msg = "Adjacency must have zero diagonal (no self-loops)"
        raise ValueError(msg)

    if not _has_hamiltonian_cycle(adj):
        msg = (
            "Motif must have a Hamiltonian cycle (a cycle visiting all nodes). "
            "This ensures sufficient connectivity for the CMA algorithm."
        )
        raise ValueError(msg)

    return adj


# Annotated type that accepts lists or arrays, converts to ndarray, then validates
AdjacencyMatrix = Annotated[
    NDArray[np.int_],
    BeforeValidator(_to_numpy),
    AfterValidator(_validate_adjacency),
]


class Motif(BaseModel):
    """A motif (subgraph) defined by its adjacency matrix.

    The motif's structural properties are derived automatically from the
    adjacency matrix. Corner types are assigned based on node degrees within
    the motif—nodes with identical degrees receive the same corner type.

    Attributes:
        adjacency: Square binary symmetric matrix defining the motif structure.
            Can be a nested list or numpy array.

    Example:
        >>> triangle = Motif(adjacency=[[0, 1, 1],
        ...                             [1, 0, 1],
        ...                             [1, 1, 0]])
        >>> triangle.num_nodes
        3
        >>> triangle.corner_types
        [1, 1, 1]
        >>> triangle.is_complete
        True
    """

    adjacency: AdjacencyMatrix

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def num_nodes(self) -> int:
        """Number of nodes in the motif."""
        return int(self.adjacency.shape[0])

    @computed_field
    @property
    def num_edges(self) -> int:
        """Number of edges in the motif."""
        return int(np.sum(self.adjacency) // 2)

    @computed_field
    @property
    def degrees(self) -> list[int]:
        """Degree of each node within the motif."""
        return [int(d) for d in np.sum(self.adjacency, axis=1)]

    @computed_field
    @property
    def corner_types(self) -> list[int]:
        """Corner type for each node (derived from degree).

        Nodes with the same degree within the motif receive the same type.
        Types are numbered starting from 1.
        """
        unique_degrees = sorted(set(self.degrees))
        degree_to_type = {d: i + 1 for i, d in enumerate(unique_degrees)}
        return [degree_to_type[d] for d in self.degrees]

    @computed_field
    @property
    def is_complete(self) -> bool:
        """True if all nodes have the same degree (all corners equivalent)."""
        return len(set(self.degrees)) == 1

    @computed_field
    @property
    def cardinalities(self) -> dict[int, int]:
        """Maps corner type to its degree within the motif.

        For a triangle, all corners have type 1 with cardinality 2
        (each corner connects to 2 other nodes in the triangle).
        """
        result: dict[int, int] = {}
        for i, corner_type in enumerate(self.corner_types):
            if corner_type not in result:
                result[corner_type] = self.degrees[i]
        return result

    @computed_field
    @property
    def type_counts(self) -> dict[int, int]:
        """Maps corner type to how many nodes have that type."""
        counts: dict[int, int] = {}
        for corner_type in self.corner_types:
            counts[corner_type] = counts.get(corner_type, 0) + 1
        return counts

    def edges_for(
        self,
        nodes: NDArray[np.int_],
    ) -> tuple[list[int], list[int]]:
        """Extract edges when this motif is instantiated with given nodes.

        Maps the motif's abstract structure onto concrete node IDs. For each
        edge (i, j) in the motif adjacency, produces edge (nodes[i], nodes[j]).

        Args:
            nodes: Array of node IDs, one per motif position. Length must
                equal num_nodes.

        Returns:
            Tuple of (sources, targets) lists for edges in this instance.

        Example:
            >>> triangle = Motif(adjacency=[[0,1,1], [1,0,1], [1,1,0]])
            >>> triangle.edges_for(np.array([10, 20, 30]))
            ([10, 10, 20], [20, 30, 30])
        """
        rows: list[int] = []
        cols: list[int] = []

        for i in range(self.num_nodes):
            for j in range(i + 1, self.num_nodes):
                if self.adjacency[i, j] == 1:
                    rows.append(int(nodes[i]))
                    cols.append(int(nodes[j]))

        return rows, cols

    def __repr__(self) -> str:
        return f"Motif(nodes={self.num_nodes}, edges={self.num_edges})"

    def __hash__(self) -> int:
        return hash(self.adjacency.tobytes())

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Motif):
            return NotImplemented
        return np.array_equal(self.adjacency, other.adjacency)
