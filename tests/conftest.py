"""Shared pytest fixtures for craeft tests."""

import numpy as np
import pytest
from scipy.sparse import csr_array


@pytest.fixture
def rng() -> np.random.Generator:
    """Seeded random generator for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def chain_graph() -> csr_array:
    """
    4-node chain graph: 0 - 1 - 2 - 3

    Adjacency:
        0  1  0  0
        1  0  1  0
        0  1  0  1
        0  0  1  0
    """
    data = np.array([1, 1, 1, 1, 1, 1], dtype=np.float64)
    rows = np.array([0, 1, 1, 2, 2, 3])
    cols = np.array([1, 0, 2, 1, 3, 2])
    return csr_array((data, (rows, cols)), shape=(4, 4))


@pytest.fixture
def complete_graph_k4() -> csr_array:
    """
    Complete graph K4 (4 nodes, all connected).

    Each node has degree 3.
    """
    n = 4
    dense = np.ones((n, n), dtype=np.float64) - np.eye(n)
    return csr_array(dense)


@pytest.fixture
def triangle_graph() -> csr_array:
    """
    Single triangle (K3): 0 - 1 - 2 - 0

    Each node has degree 2, one triangle total.
    """
    dense = np.array(
        [
            [0, 1, 1],
            [1, 0, 1],
            [1, 1, 0],
        ],
        dtype=np.float64,
    )
    return csr_array(dense)


@pytest.fixture
def star_graph() -> csr_array:
    """
    Star graph with central node 0 connected to 1, 2, 3.

    Node 0 has degree 3, others have degree 1. No triangles.
    """
    dense = np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ],
        dtype=np.float64,
    )
    return csr_array(dense)


@pytest.fixture
def two_triangles_shared_edge() -> csr_array:
    """
    Two triangles sharing edge 1-2: 0-1-2-0 and 1-2-3-1.

    Forms a "bowtie" or "diamond" shape.
    Nodes 1 and 2 are in both triangles.
    """
    dense = np.array(
        [
            [0, 1, 1, 0],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=np.float64,
    )
    return csr_array(dense)
