"""Cardinality Matching Algorithm (CMA) for network generation.

Generates networks with prescribed degree sequence and subgraph family
distribution. Supports both complete and incomplete subgraphs.

Reference:
    Ritchie et al. (2015) "Generation and analysis of networks with a
    prescribed degree sequence and subgraph family", Journal of Complex
    Networks 5(1), 1-31.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.sparse import csr_matrix

from craeft.networks.generation.configuration_model.assembly import assemble
from craeft.networks.generation.configuration_model.connectors import (
    ConnectorName,
    connect_motif,
)
from craeft.networks.generation.configuration_model.core import (
    configuration_model as simple_cm,
)
from craeft.networks.generation.configuration_model.handshake import (
    balance_allocation,
)
from craeft.networks.generation.configuration_model.matching import (
    cardinality_match,
)
from craeft.networks.generation.configuration_model.multinomial_partition import (
    multinomial_decompose,
)
from craeft.networks.generation.configuration_model.pairing import (
    pair_singles,
)
from craeft.networks.generation.configuration_model.subgraph_spec import (
    SubgraphSpec,
)


def cma(
    degrees: NDArray[np.int_],
    specs: list[SubgraphSpec],
    rng: np.random.Generator | None = None,
    connector: ConnectorName = "repeated",
) -> csr_matrix:
    """Generate a network using the Cardinality Matching Algorithm.

    Builds a network with the prescribed degree sequence and subgraph
    family distribution. For each subgraph spec, allocates hyperstubs
    via multinomial decomposition, greedily matches them to nodes, then
    connects them using the specified strategy. Remaining stubs are
    paired via the standard configuration model.

    Args:
        degrees: Per-node degree sequence (length N).
        specs: List of subgraph specifications. Each pairs a motif with
            a per-node participation sequence.
        rng: Random generator. If None, creates a default one.
        connector: Connection strategy — "repeated", "refuse", or "erased".

    Returns:
        Symmetric adjacency matrix in CSR format.

    Example:
        >>> import numpy as np
        >>> from craeft.networks.generation.configuration_model.cma import cma
        >>> from craeft.networks.generation.configuration_model.subgraph_spec import (
        ...     SubgraphSpec,
        ... )
        >>> from craeft.networks.generation.motifs import G2
        >>> rng = np.random.default_rng(42)
        >>> degrees = np.full(500, 5)
        >>> spec = SubgraphSpec(motif=G2, sequence=np.full(500, 2))
        >>> adj = cma(degrees, [spec], rng=rng)
    """
    if rng is None:
        rng = np.random.default_rng()

    if not specs:
        return simple_cm(degrees, rng)

    n = len(degrees)

    for spec in specs:
        if spec.n_nodes != n:
            msg = f"Spec sequence length {spec.n_nodes} != degree length {n}"
            raise ValueError(msg)

    # Step 1: Multinomial decomposition per spec
    decompositions = [multinomial_decompose(spec, rng) for spec in specs]

    # Step 2: Greedy cardinality matching
    allocation = cardinality_match(degrees, specs, decompositions, rng)

    # Step 3: Balance (handshake lemma)
    allocation = balance_allocation(allocation)

    # Step 4: Connect motif instances
    all_motif_rows: list[int] = []
    all_motif_cols: list[int] = []
    existing_edges: set[tuple[int, int]] = set()

    for spec_idx, spec in enumerate(specs):
        motif = spec.motif

        if motif.is_complete:
            # Flat hyperstubs for complete motifs
            corner_type = motif.corner_types[0]
            hyperstubs = allocation.node_ids_for(spec_idx, corner_type)
            rows, cols = connect_motif(
                motif,
                hyperstubs=hyperstubs,
                bins=None,
                rng=rng,
                strategy=connector,
            )
        else:
            # Typed bins for incomplete motifs
            bins = {t: allocation.node_ids_for(spec_idx, t) for t in motif.type_counts}
            rows, cols = connect_motif(
                motif,
                hyperstubs=None,
                bins=bins,
                rng=rng,
                strategy=connector,
            )

        all_motif_rows.extend(rows.tolist())
        all_motif_cols.extend(cols.tolist())

        for r, c in zip(rows.tolist(), cols.tolist()):
            existing_edges.add((min(r, c), max(r, c)))

    motif_edges = (
        np.array(all_motif_rows, dtype=np.int_),
        np.array(all_motif_cols, dtype=np.int_),
    )

    # Step 5: Pair remaining singles
    single_stubs = allocation.single_stubs()
    pairing = pair_singles(single_stubs, existing_edges, rng)

    # Step 6: Assemble
    result = assemble(n, motif_edges, pairing)
    return result.adjacency
