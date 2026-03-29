"""Base graph and subgraph abstractions for generated networks."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Self, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, computed_field
from scipy.sparse import coo_matrix, csr_matrix
from scipy.sparse.csgraph import connected_components

C = TypeVar("C", bound="GraphConfig")


@dataclass(frozen=True)
class GraphConfig:
    """Base configuration. Subclassed per graph type."""

    n: int


class BaseGraph(ABC, Generic[C]):
    """Abstract base for all generated networks.

    Stores adjacency as CSR internally. Shared interface for both
    directed and undirected graphs. Properties specific to
    directedness live on the subclasses.
    """

    def __init__(self, adjacency: csr_matrix) -> None:
        self._adjacency = adjacency

    # -- Construction --

    @classmethod
    @abstractmethod
    def from_config(cls, config: C, rng: np.random.Generator) -> Self: ...

    # -- Export --

    def to_csr(self) -> csr_matrix:
        return self._adjacency

    def to_coo(self) -> coo_matrix:
        return self._adjacency.tocoo()

    # -- Properties --

    @property
    def n_nodes(self) -> int:
        return self._adjacency.shape[0]

    @property
    @abstractmethod
    def n_edges(self) -> int: ...

    # -- Dunder --

    def __len__(self) -> int:
        return self.n_nodes

    def __repr__(self) -> str:
        return f"{type(self).__name__}(n_nodes={self.n_nodes}, n_edges={self.n_edges})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BaseGraph):
            return NotImplemented
        if self._adjacency.shape != other._adjacency.shape:
            return False
        return (self._adjacency != other._adjacency).nnz == 0


class UndirectedGraph(BaseGraph[C]):
    """Abstract base for undirected networks.

    Adjacency matrix is symmetric. Each edge counted once.
    """

    @property
    def n_edges(self) -> int:
        return self._adjacency.nnz // 2

    @property
    def degrees(self) -> NDArray[np.int_]:
        return np.asarray(self._adjacency.sum(axis=1)).flatten()

    @property
    def is_connected(self) -> bool:
        n, _ = connected_components(self._adjacency, directed=False)
        return n == 1

    @property
    @abstractmethod
    def clustering_coefficient(self) -> float: ...


class DirectedGraph(BaseGraph[C]):
    """Abstract base for directed networks.

    Adjacency matrix may be asymmetric. Edges have direction.
    """

    @property
    def n_edges(self) -> int:
        return self._adjacency.nnz

    @property
    def in_degrees(self) -> NDArray[np.int_]:
        return np.asarray(self._adjacency.sum(axis=0)).flatten()

    @property
    def out_degrees(self) -> NDArray[np.int_]:
        return np.asarray(self._adjacency.sum(axis=1)).flatten()

    @property
    def is_strongly_connected(self) -> bool:
        n, _ = connected_components(self._adjacency, directed=True, connection="strong")
        return n == 1

    @property
    def is_weakly_connected(self) -> bool:
        n, _ = connected_components(self._adjacency, directed=True, connection="weak")
        return n == 1


# ---------------------------------------------------------------------------
# Subgraph models
# ---------------------------------------------------------------------------


class Subgraph(BaseModel):
    """Undirected subgraph pattern defined by its adjacency matrix.

    Adjacency must be square, binary, symmetric, zero-diagonal,
    and contain a Hamiltonian cycle.

    Properties are derived automatically from the adjacency matrix.
    """

    adjacency: NDArray[np.int_]

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @computed_field
    @property
    def num_edges(self) -> int:
        return int(np.sum(self.adjacency) // 2)

    @computed_field
    @property
    def degrees(self) -> list[int]:
        return [int(d) for d in np.sum(self.adjacency, axis=1)]

    def __len__(self) -> int:
        return self.num_nodes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Subgraph):
            return NotImplemented
        return np.array_equal(self.adjacency, other.adjacency)

    def __repr__(self) -> str:
        return f"Subgraph(nodes={self.num_nodes}, edges={self.num_edges})"


class DirectedSubgraph(BaseModel):
    """Directed subgraph pattern. Asymmetric adjacency allowed.

    Adjacency must be square, binary, zero-diagonal,
    and contain a directed Hamiltonian cycle.
    """

    adjacency: NDArray[np.int_]

    model_config = {"arbitrary_types_allowed": True}

    @computed_field
    @property
    def num_nodes(self) -> int:
        return int(self.adjacency.shape[0])

    @computed_field
    @property
    def num_edges(self) -> int:
        return int(np.sum(self.adjacency))

    @computed_field
    @property
    def in_degrees(self) -> list[int]:
        return [int(d) for d in np.sum(self.adjacency, axis=0)]

    @computed_field
    @property
    def out_degrees(self) -> list[int]:
        return [int(d) for d in np.sum(self.adjacency, axis=1)]

    def __len__(self) -> int:
        return self.num_nodes

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, DirectedSubgraph):
            return NotImplemented
        return np.array_equal(self.adjacency, other.adjacency)

    def __repr__(self) -> str:
        return f"DirectedSubgraph(nodes={self.num_nodes}, edges={self.num_edges})"
