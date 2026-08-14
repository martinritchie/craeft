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
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import igraph as ig
import numpy as np

from craeft.graphs.base import Subgraph

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
    """Expected number of subgraph instances implied by a sequence's distribution.

    `n * mean(distribution) / subgraph.num_nodes` — total expected
    participation divided by the number of stubs consumed per instance.
    """
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
