"""SubgraphSequence: subgraph paired with a participation distribution."""

from __future__ import annotations

from dataclasses import dataclass

import igraph as ig
import numpy as np
from numpy.typing import NDArray
from scipy.stats import rv_discrete

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model.sequence.sampling import (
    _sample_sequence,
)


def _compute_orbits(adjacency: NDArray[np.int_]) -> list[int]:
    """Compute vertex orbits from the automorphism group.

    Two vertices are in the same orbit if some automorphism maps
    one to the other. Uses igraph's VF2 algorithm to enumerate
    automorphisms, then union-find to partition vertices.

    Args:
        adjacency: Square, symmetric, binary adjacency matrix.

    Returns:
        Orbit labels (consecutive integers from 0), one per vertex.
    """
    n = adjacency.shape[0]
    graph = ig.Graph.Adjacency(adjacency.tolist(), mode="undirected")
    automorphisms = graph.get_automorphisms_vf2()

    # Union-find over vertex indices
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for perm in automorphisms:
        for i, j in enumerate(perm):
            union(i, j)

    # Map roots to consecutive labels (ordered by first appearance)
    label_map: dict[int, int] = {}
    return [label_map.setdefault(find(i), len(label_map)) for i in range(n)]


def _enumerate_split_points(
    s_i: int,
    R: dict[int, int],
    orbits: list[int],
) -> list[tuple[int, ...]]:
    """Enumerate all valid orbit-assignment vectors for a node.

    A vector (c_0, ..., c_{k-1}) is valid when:
      * sum c_o = s_i  (right number of participations)
      * 0 ≤ c_o ≤ R[o]  (don't exceed what the urn has left)

    Works for any k ≥ 1 via recursion on the orbit list.
    For k ≤ 3 and moderate s_i, the number of lattice points is small.
    """
    k = len(orbits)

    if k == 1:
        o = orbits[0]
        return [(s_i,)] if s_i <= R.get(o, 0) else []

    if k == 2:
        o0, o1 = orbits[0], orbits[1]
        lo = max(0, s_i - R.get(o1, 0))
        hi = min(s_i, R.get(o0, 0))
        return [tuple(x) for x in zip(range(lo, hi + 1))] if lo <= hi else []

    # k ≥ 3: fix c_0, recurse on remaining orbits
    points: list[tuple[int, ...]] = []
    o0 = orbits[0]
    max_c0 = min(s_i, R.get(o0, 0))
    for c0 in range(max_c0 + 1):
        sub_points = _enumerate_split_points(
            s_i - c0, R, orbits[1:]
        )
        for sub in sub_points:
            points.append((c0,) + sub)

    return points


@dataclass(frozen=True)
class SubgraphSequence:
    """A subgraph paired with a participation distribution.

    Describes which subgraph structure to embed and how participation
    counts are distributed across nodes. The concrete sequence is
    sampled at generation time via ``sample``.

    Orbit properties are derived from the subgraph's automorphism
    group and cached on first access.

    Attributes:
        subgraph: The subgraph structure to embed.
        distribution: Frozen scipy discrete distribution for
            per-node participation counts (e.g. poisson(1)).
    """

    subgraph: Subgraph
    distribution: rv_discrete

    def __post_init__(self) -> None:
        # Eagerly compute orbits once — VF2 is expensive
        cached_orbits = _compute_orbits(self.subgraph.adjacency)
        object.__setattr__(self, "_cached_orbits", cached_orbits)
        object.__setattr__(
            self,
            "_cached_orbit_degrees",
            {
                o: self.subgraph.degrees[i]
                for i, o in enumerate(object.__getattribute__(self, "_cached_orbits"))
            },
        )
        counts: dict[int, int] = {}
        for o in object.__getattribute__(self, "_cached_orbits"):
            counts[o] = counts.get(o, 0) + 1
        object.__setattr__(self, "_cached_orbit_sizes", counts)
        object.__setattr__(
            self,
            "_cached_is_vertex_transitive",
            len(set(object.__getattribute__(self, "_cached_orbits"))) == 1,
        )

    def sample(self, n: int, rng: np.random.Generator) -> NDArray[np.int_]:
        """Sample a participation sequence of length n.

        Rejection samples until all values are non-negative and the
        total is divisible by the subgraph's node count. Values may
        exceed n - 1 (degree budget enforcement is deferred to the
        allocation step).

        Args:
            n: Number of nodes in the network.
            rng: Random number generator.

        Returns:
            Array of n non-negative counts whose sum is divisible
            by subgraph.num_nodes.
        """
        return _sample_sequence(
            n,
            self.distribution,
            rng,
            divisor=self.subgraph.num_nodes,
            max_value=n,
        )

    def _split_by_orbit(
        self,
        sequence: NDArray[np.int_],
        rng: np.random.Generator,
    ) -> dict[int, NDArray[np.int_]]:
        """Split a participation sequence into per-orbit counts.

        For vertex-transitive subgraphs (single orbit), returns the
        sequence unchanged. For non-transitive subgraphs, uses
        sequential conditional sampling: each node draws its orbit
        assignments from a shared urn, ensuring global totals match
        the subgraph's orbit proportions exactly.

        The urn is initialised with M * σ_o balls of each orbit o,
        where M = total participations / |V(H)|. Nodes are processed
        in descending participation order so that high-degree nodes
        (with the tightest constraints) are resolved first.

        For k=2 orbits, a closed-form solution samples x_i =
        C_{i,0} uniformly from [ℓ_i, u_i] per node. For k≥3,
        enumerates lattice points of the local polytope.

        Args:
            sequence: A sampled participation sequence from ``sample``.
            rng: Random number generator.

        Returns:
            Mapping from orbit label to per-node count array.

        Raises:
            ValueError: If the participation sequence is incompatible
                with the subgraph's orbit proportions.
        """
        if self.is_vertex_transitive:
            return {0: sequence.copy()}

        sizes = self.orbit_sizes
        n = len(sequence)
        k = len(sizes)

        total_parts = int(sequence.sum())
        num_instances = total_parts // self.subgraph.num_nodes

        # Initialise urn: R_o = M * σ_o for each orbit
        R = {o: num_instances * sizes[o] for o in range(k)}

        result = {o: np.zeros(n, dtype=np.int_) for o in range(k)}

        # Process nodes in descending participation order
        order = np.argsort(-sequence)

        if k == 2:
            for idx in order:
                s_i = int(sequence[idx])
                if s_i == 0:
                    break
                # Bounds for x_i (orbit-0 count):
                # ℓ_i = max(0, s_i − R₁) — forced by remaining leaf slots
                # u_i = min(s_i, R₀) — capped by remaining hub slots
                lo = max(0, s_i - R[1])
                hi = min(s_i, R[0])
                if lo > hi:
                    msg = (
                        f"Infeasible split at node {idx}: "
                        f"s={s_i}, R₀={R[0]}, R₁={R[1]}, "
                        f"bounds=[{lo}, {hi}]"
                    )
                    raise ValueError(msg)
                x_i = int(rng.integers(lo, hi + 1))
                result[0][idx] = x_i
                result[1][idx] = s_i - x_i
                R[0] -= x_i
                R[1] -= s_i - x_i
        else:
            for idx in order:
                s_i = int(sequence[idx])
                if s_i == 0:
                    break
                feasible = _enumerate_split_points(
                    s_i, R, list(range(k))
                )
                if not feasible:
                    msg = (
                        f"Infeasible split at node {idx}: "
                        f"s={s_i}, R={R}, no valid lattice point"
                    )
                    raise ValueError(msg)
                choice = feasible[int(rng.integers(len(feasible)))]
                for o, c_o in enumerate(choice):
                    result[o][idx] = c_o
                    R[o] -= c_o

        return result

    @property
    def orbits(self) -> list[int]:
        """Orbit label for each node in the subgraph.

        Nodes in the same automorphism orbit receive the same label.
        Labels are consecutive integers starting from 0, assigned
        in order of first vertex appearance.
        """
        return self._cached_orbits  # type: ignore[attr-defined]

    @property
    def orbit_degrees(self) -> dict[int, int]:
        """Maps orbit label to its degree within the subgraph.

        Well-defined because all vertices in an orbit share the
        same degree (automorphisms preserve adjacency).
        """
        return self._cached_orbit_degrees  # type: ignore[attr-defined]

    @property
    def orbit_sizes(self) -> dict[int, int]:
        """Maps orbit label to how many nodes belong to that orbit."""
        return self._cached_orbit_sizes  # type: ignore[attr-defined]

    @property
    def is_vertex_transitive(self) -> bool:
        """True if all nodes belong to a single orbit."""
        return self._cached_is_vertex_transitive  # type: ignore[attr-defined]

    def edges_for(self, nodes: NDArray[np.int_]) -> tuple[list[int], list[int]]:
        """Map the subgraph's structure onto concrete node IDs.

        Given ``nodes`` (length = subgraph.num_nodes), emits one edge
        per existing edge in the subgraph, mapping each position to the
        concrete node at that index via ``nodes[position]``.

        Only traverses the upper triangle of the adjacency to avoid
        double-counting edges.
        """
        rows, cols = np.triu_indices(self.subgraph.num_nodes, k=1)
        mask = self.subgraph.adjacency[rows, cols] > 0
        rows, cols = rows[mask], cols[mask]
        return (nodes[rows].tolist(), nodes[cols].tolist())
