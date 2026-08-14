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
        orbit_counts: Prescribed per-node counts per orbit. When
            given, overrides sampling — ``_split_by_orbit`` returns
            it unchanged instead of drawing from the urn. Validated
            in ``__post_init__``: keys must match the orbit labels
            exactly, all arrays must be equal length and
            non-negative, and each orbit's total must equal
            ``M * orbit_sizes[o]`` for a common integer ``M`` across
            all orbits (the invariant the urn sampler otherwise
            enforces implicitly).
    """

    subgraph: Subgraph
    distribution: rv_discrete
    orbit_counts: dict[int, NDArray[np.int_]] | None = None

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
        if self.orbit_counts is not None:
            self._validate_orbit_counts()

    def _validate_orbit_counts(self) -> None:
        """Validate a prescribed per-orbit decomposition.

        Raises:
            ValueError: If keys don't match the orbit labels, arrays
                have mismatched lengths or negative entries, or
                per-orbit totals aren't consistent with forming a
                common integer number of whole subgraph instances.
        """
        counts = self.orbit_counts
        assert counts is not None  # guarded by caller
        sizes = object.__getattribute__(self, "_cached_orbit_sizes")

        expected_keys = set(sizes)
        got_keys = set(counts)
        if got_keys != expected_keys:
            msg = (
                f"orbit_counts keys {sorted(got_keys)} do not match "
                f"orbit labels {sorted(expected_keys)}"
            )
            raise ValueError(msg)

        lengths = {o: len(arr) for o, arr in counts.items()}
        if len(set(lengths.values())) > 1:
            msg = f"orbit_counts arrays have mismatched lengths: {lengths}"
            raise ValueError(msg)

        for o, arr in counts.items():
            if np.any(np.asarray(arr) < 0):
                msg = f"orbit_counts[{o}] contains negative counts"
                raise ValueError(msg)

        totals = {o: int(np.asarray(arr).sum()) for o, arr in counts.items()}
        multiples: set[int] = set()
        for o, total in totals.items():
            size_o = sizes[o]
            if total % size_o != 0:
                msg = (
                    f"orbit_counts[{o}] total {total} is not a multiple "
                    f"of orbit size sigma_{o}={size_o} — cannot form "
                    "whole subgraph instances"
                )
                raise ValueError(msg)
            multiples.add(total // size_o)

        if len(multiples) > 1:
            msg = (
                "orbit_counts totals imply inconsistent instance counts "
                f"across orbits (no common M): totals={totals}, "
                f"sizes={sizes}, implied M values={sorted(multiples)}"
            )
            raise ValueError(msg)

    def sample(
        self,
        n: int,
        rng: np.random.Generator,
        max_per_node: NDArray[np.int_] | None = None,
    ) -> NDArray[np.int_]:
        """Sample a participation sequence of length n.

        Rejection samples until all values are non-negative and the
        total is divisible by the subgraph's node count. Values may
        exceed n - 1 unless ``max_per_node`` is given.

        Args:
            n: Number of nodes in the network.
            rng: Random number generator.
            max_per_node: Optional length-n array of per-node upper
                bounds (ticket 003). Without this, participation is
                sampled with no reference to each node's degree
                budget, so a build's success is down to luck rather
                than construction — see ``ConfigModelGraph.from_config``,
                which passes a conservative degree-derived cap here.
                When given, over-cap values are clipped (not
                rejected) — see ``_sample_sequence`` for the bias this
                introduces.

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
            max_per_node=max_per_node,
        )

    def _split_by_orbit(
        self,
        sequence: NDArray[np.int_],
        rng: np.random.Generator,
    ) -> dict[int, NDArray[np.int_]]:
        """Split a participation sequence into per-orbit counts.

        When ``orbit_counts`` is prescribed, returns it unchanged —
        sampling is bypassed entirely, including for vertex-transitive
        subgraphs. Otherwise, for vertex-transitive subgraphs (single
        orbit), returns the sequence unchanged. For non-transitive
        subgraphs, uses sequential conditional sampling: each node
        draws its orbit assignments from a shared urn, ensuring global
        totals match the subgraph's orbit proportions exactly.

        The urn is initialised with M * σ_o balls of each orbit o,
        where M = total participations / |V(H)|. Nodes are processed
        in descending participation order so that high-degree nodes
        (with the tightest constraints) are resolved first.

        For k=2 orbits, a closed-form solution samples x_i =
        C_{i,0} uniformly from [ℓ_i, u_i] per node. For k≥3,
        enumerates lattice points of the local polytope.

        Args:
            sequence: A sampled participation sequence from ``sample``.
                Ignored when ``orbit_counts`` is prescribed.
            rng: Random number generator. Ignored when ``orbit_counts``
                is prescribed.

        Returns:
            Mapping from orbit label to per-node count array.

        Raises:
            ValueError: If the participation sequence is incompatible
                with the subgraph's orbit proportions.
        """
        if self.orbit_counts is not None:
            return self.orbit_counts

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


def split_deterministic(
    sequence: NDArray[np.int_],
    orbits: SubgraphSequence,
) -> dict[int, NDArray[np.int_]]:
    """Largest-remainder split of participations across orbits.

    Allocates each node's participations to orbits in fixed
    proportion — matching the subgraph's orbit sizes — using the
    largest-remainder (Hamilton) apportionment method. This is fully
    deterministic (no random draws), so nodes with equal
    participation counts receive equal orbit counts wherever
    divisibility allows.

    Orbits are processed in label order. At each step (all but the
    last orbit), the orbit's target count is met *exactly* via
    largest-remainder rounding of each node's remaining capacity.
    The last orbit is never rounded — it simply absorbs whatever
    capacity remains, which is guaranteed (by construction) to equal
    exactly its own target. So every orbit's column sum, not just
    the rounded ones, comes out exact — the output is always valid
    as ``SubgraphSequence.orbit_counts``.

    Args:
        sequence: Per-node participation counts. Sum must be
            divisible by ``orbits.subgraph.num_nodes``.
        orbits: Subgraph sequence describing the orbit structure
            (sizes and vertex count) to split against.

    Returns:
        Mapping from orbit label to per-node count array.
    """
    if orbits.is_vertex_transitive:
        return {0: sequence.copy()}

    sizes = orbits.orbit_sizes
    n = len(sequence)
    total_parts = int(sequence.sum())
    num_instances = total_parts // orbits.subgraph.num_nodes

    remaining_capacity = sequence.astype(np.int_).copy()
    orbit_labels = sorted(sizes)
    result: dict[int, NDArray[np.int_]] = {}

    remaining_h = orbits.subgraph.num_nodes
    for pos, o in enumerate(orbit_labels):
        target = num_instances * sizes[o]

        if pos == len(orbit_labels) - 1:
            # Last orbit: whatever remains is forced, and exactly
            # matches `target` by construction.
            result[o] = remaining_capacity.copy()
            continue

        # Exact-integer largest remainder: numer/remaining_h is the
        # ideal (real-valued) share, computed without floats to avoid
        # precision artefacts.
        numer = remaining_capacity.astype(np.int64) * sizes[o]
        floor_counts = (numer // remaining_h).astype(np.int_)
        remainder = (numer % remaining_h).astype(np.int_)
        deficit = target - int(floor_counts.sum())

        counts = floor_counts.copy()
        if deficit > 0:
            # Largest remainder first; ties broken by node index for
            # determinism.
            order = np.lexsort((np.arange(n), -remainder))
            for idx in order[:deficit]:
                counts[idx] += 1

        result[o] = counts
        remaining_capacity = remaining_capacity - counts
        remaining_h -= sizes[o]

    return result


def split_by_degree_rank(
    sequence: NDArray[np.int_],
    degrees: NDArray[np.int_],
    orbits: SubgraphSequence,
) -> dict[int, NDArray[np.int_]]:
    """Assign high-cardinality orbits to high-degree nodes.

    The "push clustered subgraphs onto hubs" construction (2017
    paper, §3.3): ranks orbits by their within-subgraph degree
    (descending — the hub role first) and nodes by their network
    degree (descending), then greedily fills each orbit's target
    count from the highest-remaining-degree nodes with participation
    capacity left. Network hubs preferentially fill hub-like orbit
    roles, concentrating designed clustering on high-degree nodes.

    Deterministic given ``degrees`` — no random draws. Ties in
    ``degrees`` are broken by node index (stable sort) and ties in
    orbit degree are broken by orbit label.

    Args:
        sequence: Per-node participation counts. Sum must be
            divisible by ``orbits.subgraph.num_nodes``.
        degrees: Per-node network degree, same length as
            ``sequence``. Used only to rank priority — this does
            *not* check the degree budget; that is
            ``allocate_subgraphs``'s job.
        orbits: Subgraph sequence describing the orbit structure.

    Returns:
        Mapping from orbit label to per-node count array.
    """
    if orbits.is_vertex_transitive:
        return {0: sequence.copy()}

    sizes = orbits.orbit_sizes
    orbit_degrees = orbits.orbit_degrees
    n = len(sequence)
    total_parts = int(sequence.sum())
    num_instances = total_parts // orbits.subgraph.num_nodes

    remaining_capacity = sequence.astype(np.int_).copy()
    result = {o: np.zeros(n, dtype=np.int_) for o in sizes}

    # Highest within-subgraph degree (hub-like orbits) first.
    orbit_order = sorted(sizes, key=lambda o: (-orbit_degrees[o], o))
    # Highest network degree first; stable so equal-degree nodes
    # keep their original relative order.
    node_order = np.argsort(-degrees, kind="stable")

    for o in orbit_order:
        need = num_instances * sizes[o]
        if need == 0:
            continue
        for idx in node_order:
            if need <= 0:
                break
            take = min(int(remaining_capacity[idx]), need)
            if take <= 0:
                continue
            result[o][idx] += take
            remaining_capacity[idx] -= take
            need -= take

    return result
