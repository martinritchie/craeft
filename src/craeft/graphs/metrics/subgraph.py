"""Designed clustering: closed-form clustering metrics from a config, before generation.

Global clustering is fully determined by a `ConfigModelConfig` before any graph is
built — the degree sequence pins the denominator (connected triples) and the
subgraph sequences pin the numerator (expected triangles). This module exposes
that closed form so callers can design a target clustering (or a matched pair of
configs) without paying for generation.

The realized clustering on a generated graph will differ from the designed value
by a "by-product" term: triangles formed incidentally by random single-stub
pairing, plus any overlap between subgraph instances. See
`ConfigModelGraph.designed_clustering` for comparing designed vs realized.

`mean_excess_degree` and `predicted_cycle_floor` give the other half of that
picture in closed form: how much cycle structure a configuration-model null
produces on its own, from the degree sequence alone, before any subgraph is
designed in. Together they say what a measured cycle count has to beat before
it counts as designed structure.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import igraph as ig
import numpy as np
from numpy.typing import NDArray

from craeft.graphs.base import Subgraph
from craeft.graphs.metrics.cycles import MIN_CYCLE_LENGTH

if TYPE_CHECKING:
    from craeft.graphs.configuration_model.models import ConfigModelConfig
    from craeft.graphs.configuration_model.sequence.subgraph_sequence import (
        SubgraphSequence,
    )


def unique_triangles(subgraph: Subgraph) -> int:
    """Number of distinct triangles contained in a subgraph pattern.

    Counts 3-cliques in the subgraph's adjacency matrix. This is the
    per-instance triangle contribution used by `designed_triangles`.

    Args:
        subgraph: Subgraph pattern to inspect.

    Returns:
        Count of triangles (3-cliques) in the subgraph.

    Example:
        >>> import numpy as np
        >>> from craeft.graphs.base import Subgraph
        >>> triangle = Subgraph(adjacency=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]))
        >>> unique_triangles(triangle)
        1
    """
    g = ig.Graph.Adjacency((subgraph.adjacency > 0).tolist(), mode="undirected")
    g.simplify()
    return len(g.cliques(min=3, max=3))


def triangles_per_orbit(sequence: SubgraphSequence) -> dict[int, int]:
    """Triangles incident to a node at each orbit position, within one instance.

    For each automorphism orbit of the subgraph, counts how many triangles a
    node occupying that orbit position participates in. Orbit-mates always
    have identical triangle counts (automorphisms preserve adjacency), so one
    representative per orbit determines the whole mapping.

    Args:
        sequence: Subgraph sequence whose orbit structure to inspect.

    Returns:
        Mapping from orbit label to triangle count per node in that orbit.

    Example:
        Diamond (K4 minus one edge): tips participate in 1 triangle, hubs in 2.
    """
    adjacency = sequence.subgraph.adjacency
    g = ig.Graph.Adjacency((adjacency > 0).tolist(), mode="undirected")
    g.simplify()
    triangles = g.cliques(min=3, max=3)

    result: dict[int, int] = {}
    for node, orbit in enumerate(sequence.orbits):
        if orbit not in result:
            result[orbit] = sum(1 for tri in triangles if node in tri)
    return result


def _expected_instances(sequence: SubgraphSequence, n: int) -> float:
    """Number of subgraph instances implied by a sequence.

    When the sequence is prescribed (ticket 007), the instance count is
    not an expectation at all: `orbit_counts` pins it exactly, and its
    validation has already established a common `M` across orbits. Use
    it — the designed figures downstream become exact rather than
    correct-in-expectation, and `n` is irrelevant because the
    prescription already names every node.

    Otherwise, `n * mean(distribution) / subgraph.num_nodes` — total
    expected participation divided by the stubs consumed per instance.
    """
    if sequence.num_instances is not None:
        return float(sequence.num_instances)
    assert sequence.distribution is not None  # enforced in __post_init__
    return n * sequence.distribution.mean() / sequence.subgraph.num_nodes


def designed_triangles(config: ConfigModelConfig) -> float:
    """Expected number of designed triangles implied by a config's subgraph sequences.

    Closed-form: `sum_j M_j * u_j`, where `M_j` is the expected instance count
    of subgraph sequence `j` and `u_j` is its unique triangles per instance
    (`unique_triangles`). Computed entirely from the config, before generation.

    This is the *designed* triangle count only. It does not include triangles
    formed incidentally by random single-stub pairing or by overlap between
    subgraph instances — the realized count on a generated graph will be this
    plus that by-product floor.

    Args:
        config: Configuration model configuration.

    Returns:
        Expected count of designed triangles. May be fractional since it is
        an expectation over the participation distributions. Zero if there
        are no subgraph sequences.
    """
    designed = 0.0
    for seq in config.sequences:
        m = _expected_instances(seq, config.n)
        designed += m * unique_triangles(seq.subgraph)
    return designed


def designed_clustering(config: ConfigModelConfig) -> float:
    """Global clustering coefficient implied by a config, before generation.

    Exact for the designed structure only:

        C = 3 * designed_triangles(config) / triples

    where `triples = sum_i C(k_i, 2)` depends only on the degree sequence.

    The realized clustering on a generated graph will be this plus a
    by-product term (random triangles from single-stub pairing, plus overlap
    between subgraph instances) — see `designed_triangles` and
    `ConfigModelGraph.designed_clustering`.

    Args:
        config: Configuration model configuration.

    Returns:
        Designed clustering coefficient. 0.0 if the degree sequence has no
        connected triples, or if no subgraph sequences are specified.
    """
    k = np.asarray(config.degrees)
    triples = int((k * (k - 1) // 2).sum())
    if triples == 0:
        return 0.0
    return 3 * designed_triangles(config) / triples


def mean_excess_degree(degrees: NDArray[np.int_]) -> float:
    """The configuration model's branching factor, kappa = <k(k-1)>/<k>.

    The expected number of *further* edges reachable from a node arrived at by
    following a random edge. It is the quantity that governs how much cycle
    structure a configuration-model null throws off by accident, which is why
    `predicted_cycle_floor` is a function of it alone.

    For a regular sequence with constant degree k this is exactly `k - 1`.

    Args:
        degrees: Degree sequence.

    Returns:
        Mean excess degree. 0.0 if the sequence has no stubs at all.

    Example:
        >>> import numpy as np
        >>> float(mean_excess_degree(np.full(100, 5)))
        4.0
    """
    k = np.asarray(degrees, dtype=np.float64)
    mean_degree = k.mean() if k.size else 0.0
    if mean_degree == 0:
        return 0.0
    return float((k * (k - 1)).mean() / mean_degree)


def predicted_cycle_floor(degrees: NDArray[np.int_], length: int) -> float:
    """Expected L-cycles in a configuration model null: kappa**L / (2L).

    The standard configuration-model result, with `kappa` the mean excess
    degree (`mean_excess_degree`). Computed from the degree sequence alone —
    no graph is generated — so it gives the by-product floor a measured cycle
    count has to clear before it evidences *designed* structure.

    Counts all cycles; the induced-cycle floor is slightly lower, with the
    gap growing with density (measured ~5-15% for hexagons at kappa=9).
    Asymptotic in n; finite-sample kappa is used, so a heavy-tailed sequence
    reports its actual (cutoff-dependent) floor rather than a diverged one.

    Args:
        degrees: Degree sequence of the null model.
        length: Cycle length, at least 3.

    Returns:
        Expected number of cycles of that length. Never negative.

    Raises:
        ValueError: If `length` is less than 3.

    Example:
        >>> import numpy as np
        >>> predicted_cycle_floor(np.full(100, 5), 4)
        32.0
    """
    if length < MIN_CYCLE_LENGTH:
        raise ValueError(f"Cycle length must be at least {MIN_CYCLE_LENGTH}")
    kappa = mean_excess_degree(degrees)
    if kappa <= 0:
        return 0.0
    return kappa**length / (2 * length)
