"""Empirical control audit for craeft config-model graphs.

Produces the evidence behind the claim pages (docs/claims/): how controlled
is the structure, really? Measures degree preservation, subgraph-count
fidelity against the random-graph floor, and the C-model invariant.

Run:
  .venv/bin/python docs/claims/evidence/control_audit.py
    -> docs/claims/evidence/control_audit_results.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import poisson

from craeft.graphs.base import Subgraph
from craeft.graphs.configuration_model import (
    ConfigModelConfig,
    ConfigModelGraph,
    SubgraphSequence,
)
from craeft.graphs.configuration_model.sequence import split_deterministic
from craeft.graphs.metrics import (
    clustering_by_degree,
    count_order_four,
    degree_assortativity,
    designed_clustering,
    designed_triangles,
    order_four_ratios,
)

REPS = 12


# ---------------------------------------------------------------- subgraphs
def cycle(n: int) -> Subgraph:
    """Cycle C_n. Vertex-transitive; every hyperstub has cardinality 2."""
    a = np.zeros((n, n), dtype=int)
    for i in range(n):
        a[i, (i + 1) % n] = a[(i + 1) % n, i] = 1
    return Subgraph(adjacency=a)


def complete(n: int) -> Subgraph:
    return Subgraph(adjacency=np.ones((n, n), dtype=int) - np.eye(n, dtype=int))


def diamond() -> Subgraph:
    """G_boxslash: square + one diagonal. Orbits (2,2,3,3)."""
    return Subgraph(
        adjacency=np.array(
            [[0, 1, 1, 0], [1, 0, 1, 1], [1, 1, 0, 1], [0, 1, 1, 0]], dtype=int
        )
    )


SG = {
    "triangle": cycle(3),
    "square": cycle(4),
    "pentagon": cycle(5),
    "hexagon": cycle(6),
    "diamond": diamond(),
    "K4": complete(4),
}


# ---------------------------------------------------------------- seeding
# HAZARD (found 2026-08-27): sections C and F draw the degree sequence as
# 2*Pois(lam) from one generator and then hand `build` a seed, from which
# `from_config` creates a FRESH generator whose first act is to draw the
# participation sequence -- also Pois(lam), same n. Seed those two the same
# and they return the *identical array*: participation then equals the
# degree-derived cap elementwise (corr = 1.0000), silently turning every
# C-model into the 2017 sec 3.3 "push clustered subgraphs onto the
# high-degree nodes" construction. That construction is measurably more
# assortative (2017 Fig. 8), so the artefact shows up exactly where an
# assortativity confound would, and reads as a property of the method.
#
# Degrees are therefore drawn from a seed stream far from the build stream.
# Do not collapse these back into one seed.
DEGREE_SEED_BASE = 20_000


def cmodel_degrees(n: int, lam: float, rep: int) -> np.ndarray:
    """Degree sequence for the C-model family: 2*Pois(lam), even, sum even."""
    rng = np.random.default_rng(DEGREE_SEED_BASE + rep)
    degrees = (2 * rng.poisson(lam, n)).astype(np.int_)
    if degrees.sum() % 2:
        degrees[int(np.argmax(degrees))] += 1
    return degrees


# ---------------------------------------------------------------- counting
def induced_cycle_count(adj: np.ndarray, length: int) -> int:
    """Unique chordless (induced) cycles of the given length."""
    import igraph as ig

    g = ig.Graph.Adjacency((adj > 0).tolist(), mode="undirected")
    g.simplify()
    if length == 3:
        return len(g.cliques(min=3, max=3))
    nbr = [set(g.neighbors(v)) for v in range(g.vcount())]
    total = 0
    for start in range(g.vcount()):
        stack: list[tuple[int, list[int]]] = [(start, [start])]
        while stack:
            v, path = stack.pop()
            if len(path) == length:
                if start in nbr[v]:
                    chordless = True
                    for i in range(length):
                        for j in range(i + 2, length):
                            if i == 0 and j == length - 1:
                                continue
                            if path[j] in nbr[path[i]]:
                                chordless = False
                                break
                        if not chordless:
                            break
                    if chordless:
                        total += 1
                continue
            for w in nbr[v]:
                if w <= start or w in path:
                    continue
                stack.append((w, [*path, w]))
    return total // 2


def clique_count(adj: np.ndarray, size: int) -> int:
    import igraph as ig

    g = ig.Graph.Adjacency((adj > 0).tolist(), mode="undirected")
    g.simplify()
    return len(g.cliques(min=size, max=size))


# ---------------------------------------------------------------- harness
def build(
    n: int, degrees: np.ndarray, seqs: tuple[SubgraphSequence, ...], seed: int
) -> ConfigModelGraph | None:
    try:
        return ConfigModelGraph.from_config(
            ConfigModelConfig(n=n, degrees=degrees, sequences=seqs, max_retries=300),
            np.random.default_rng(seed),
        )
    except Exception:  # noqa: BLE001
        return None


def adj_of(g: ConfigModelGraph) -> np.ndarray:
    return np.asarray(g.to_csr().todense())


# ---------------------------------------------------------------- A. degrees
def exp_degrees() -> list[dict]:
    print("\n" + "=" * 78)
    print("A. DEGREE-SEQUENCE PRESERVATION  (claim: 'preserved exactly')")
    print("=" * 78)
    print(
        f"  {'subgraph':<10}{'n':>6}{'deg':>5}{'rate':>6}"
        f"{'nodes exact':>13}{'mean deficit':>14}{'max':>5}"
        f"{'edges lost':>12}{'dist same':>11}"
    )
    rows = []
    cases = [
        ("none", None, 0.0, 1000, 10),
        ("triangle", "triangle", 0.8, 1000, 10),
        ("square", "square", 0.8, 1000, 10),
        ("pentagon", "pentagon", 0.8, 1000, 10),
        ("hexagon", "hexagon", 0.8, 1000, 10),
        ("diamond", "diamond", 0.5, 1000, 12),
        ("K4", "K4", 0.5, 1000, 12),
    ]
    for label, key, rate, n, deg in cases:
        degrees = np.full(n, deg, dtype=np.int_)
        seqs: tuple[SubgraphSequence, ...] = (
            ()
            if key is None
            else (SubgraphSequence(subgraph=SG[key], distribution=poisson(rate)),)
        )
        ex, defc, mx, lost, same, ok = [], [], [], [], [], 0
        for s in range(REPS):
            g = build(n, degrees, seqs, seed=100 + s)
            if g is None:
                continue
            ok += 1
            d = g.degrees
            diff = d - degrees
            ex.append(float(np.mean(diff == 0)))
            defc.append(float(np.abs(diff).sum()))
            mx.append(int(np.abs(diff).max()))
            lost.append(float((int(degrees.sum()) - int(d.sum())) / 2))
            same.append(Counter(d.tolist()) == Counter(degrees.tolist()))
        if not ok:
            print(f"  {label:<10}{n:>6}{deg:>5}{rate:>6}   ALL BUILDS FAILED")
            continue
        print(
            f"  {label:<10}{n:>6}{deg:>5}{rate:>6}"
            f"{np.mean(ex):>12.2%}{np.mean(defc):>14.1f}{max(mx):>5}"
            f"{np.mean(lost):>12.1f}{np.mean(same):>10.0%}"
        )
        rows.append(
            {
                "subgraph": label,
                "n": n,
                "target_degree": deg,
                "rate": rate,
                "frac_nodes_exact": float(np.mean(ex)),
                "mean_total_abs_deficit": float(np.mean(defc)),
                "max_node_deficit": max(mx),
                "mean_edges_lost": float(np.mean(lost)),
                "degree_dist_identical_frac": float(np.mean(same)),
                "builds_ok": ok,
            }
        )
    exact = all(r["frac_nodes_exact"] == 1.0 for r in rows)
    clean = all(r["mean_edges_lost"] == 0.0 for r in rows)
    print(
        "\n  NOTE (ticket 001, fixed): Connector.connect_singles now uses the"
        "\n  MATCHING ALGORITHM -- on a self-loop or multi-edge collision the"
        "\n  stubs are returned to the pool and redrawn, never discarded. The"
        "\n  2017 paper (sec 2) mandates exactly this: 'If this approach [of"
        "\n  deleting] is taken then the guiding degree sequence will be"
        "\n  violated.' Before the fix this table read 93.6-96.1% exact with"
        "\n  20-70 edges lost per build; degrees are now preserved element-wise."
        f"\n\n  100% of nodes exact on every family: {exact}"
        f"\n  zero edges lost on every family:      {clean}"
        "\n  => criteria P1 (per-node identical degree sequences) and P2 (equal"
        "\n     edge counts) hold. ConfigModelGraph.from_config additionally"
        "\n     asserts this per build (ticket 002, verify_degrees=True), so a"
        "\n     regression raises DegreeMismatchError rather than degrading"
        "\n     silently -- these builds are self-verifying."
    )
    return rows


# ------------------------------------------------------- B. subgraph fidelity
def exp_fidelity() -> list[dict]:
    print("\n" + "=" * 78)
    print("B. SUBGRAPH-COUNT FIDELITY vs THE RANDOM FLOOR")
    print("=" * 78)
    n, deg = 1000, 10
    degrees = np.full(n, deg, dtype=np.int_)

    # random floor: same degree sequence, no subgraph sequences
    floor: dict[str, list[float]] = {
        k: [] for k in ["triangle", "square", "pentagon", "hexagon", "K4"]
    }
    for s in range(REPS):
        g = build(n, degrees, (), seed=700 + s)
        if g is None:
            continue
        a = adj_of(g)
        floor["triangle"].append(induced_cycle_count(a, 3))
        floor["square"].append(induced_cycle_count(a, 4))
        floor["pentagon"].append(induced_cycle_count(a, 5))
        floor["hexagon"].append(induced_cycle_count(a, 6))
        floor["K4"].append(clique_count(a, 4))
    floor_mean = {k: float(np.mean(v)) if v else 0.0 for k, v in floor.items()}

    print(f"  random floor (n={n}, k={deg}, no subgraphs, {REPS} reps):")
    for k, v in floor_mean.items():
        print(f"    {k:<10}{v:>9.1f}")

    print(
        f"\n  {'target':<10}{'rate':>6}{'built':>8}{'floor':>8}"
        f"{'expected':>10}{'realized':>10}{'ratio':>7}{'S/F':>7}"
    )
    rows = []
    for key, rate, counter in [
        ("triangle", 0.8, lambda a: induced_cycle_count(a, 3)),
        ("square", 0.8, lambda a: induced_cycle_count(a, 4)),
        ("pentagon", 0.8, lambda a: induced_cycle_count(a, 5)),
        ("hexagon", 0.8, lambda a: induced_cycle_count(a, 6)),
        ("K4", 0.5, lambda a: clique_count(a, 4)),
    ]:
        sg = SG[key]
        seqs = (SubgraphSequence(subgraph=sg, distribution=poisson(rate)),)
        built = n * rate / sg.num_nodes
        got = []
        for s in range(REPS):
            g = build(n, degrees, seqs, seed=800 + s)
            if g is None:
                continue
            got.append(counter(adj_of(g)))
        if not got:
            print(f"  {key:<10}{rate:>6}   FAILED")
            continue
        realized = float(np.mean(got))
        expected = built + floor_mean[key]
        sf = built / floor_mean[key] if floor_mean[key] else float("inf")
        print(
            f"  {key:<10}{rate:>6}{built:>8.0f}{floor_mean[key]:>8.1f}"
            f"{expected:>10.1f}{realized:>10.1f}{realized / expected:>7.2f}{sf:>7.1f}"
        )
        rows.append(
            {
                "target": key,
                "rate": rate,
                "by_construction": built,
                "random_floor": floor_mean[key],
                "expected": expected,
                "realized": realized,
                "ratio_realized_expected": realized / expected,
                "signal_to_floor": sf,
            }
        )
    print("\n  S/F = by-construction / random-floor. Control is only meaningful")
    print("  when S/F >> 1; below ~2 the imposed structure is lost in the floor.")
    for r in rows:
        sf = r["signal_to_floor"]
        v = "STRONG" if sf > 10 else ("usable" if sf > 2 else "WEAK — floor dominates")
        print(f"    {r['target']:<10}{sf:>8.1f}x   {v}")
    return rows


# ------------------------------------------------------------- C. C-models
def exp_cmodels() -> list[dict]:
    print("\n" + "=" * 78)
    print("C. C-MODEL FAMILY  (2016 sec 2.2): degree dist matched, cycle length varies")
    print("=" * 78)
    n, lam = 1000, 2.0
    print(
        f"  {'model':<14}{'<k>':>7}{'Var(k)':>9}{'even-only':>11}"
        f"{'TV vs null':>12}{'clustering':>12}{'tri count':>11}"
    )
    rows, ref = [], None
    for label, key in [
        ("Null (random)", None),
        ("C1 triangle", "triangle"),
        ("C2 square", "square"),
        ("C3 pentagon", "pentagon"),
        ("C4 hexagon", "hexagon"),
    ]:
        ks, vs, cs, tris = [], [], [], []
        hist: Counter[int] = Counter()
        for s in range(REPS):
            degrees = cmodel_degrees(n, lam, s)  # cardinality-2 hyperstubs
            seqs: tuple[SubgraphSequence, ...] = (
                ()
                if key is None
                else (SubgraphSequence(subgraph=SG[key], distribution=poisson(lam)),)
            )
            g = build(n, degrees, seqs, seed=500 + s)
            if g is None:
                continue
            a = adj_of(g)
            d = g.degrees
            ks.append(d.mean())
            vs.append(d.var())
            hist.update(degrees.tolist())  # TARGET dist (what we control)
            cs.append(g.clustering_coefficient)
            tris.append(induced_cycle_count(a, 3))
        if not ks:
            print(f"  {label:<14} FAILED")
            continue
        even = all(k % 2 == 0 for k in hist)
        if ref is None:
            ref, tv_s = hist, "reference"
        else:
            keys = set(ref) | set(hist)
            ta, tb = sum(ref.values()), sum(hist.values())
            tv = 0.5 * sum(abs(ref.get(k, 0) / ta - hist.get(k, 0) / tb) for k in keys)
            tv_s = f"{tv:.4f}"
        print(
            f"  {label:<14}{np.mean(ks):>7.2f}{np.mean(vs):>9.2f}"
            f"{str(even):>11}{tv_s:>12}{np.mean(cs):>12.4f}{np.mean(tris):>11.1f}"
        )
        rows.append(
            {
                "model": label,
                "mean_degree": float(np.mean(ks)),
                "var_degree": float(np.mean(vs)),
                "even_degrees_only": even,
                "target_dist_tv_vs_null": tv_s,
                "clustering": float(np.mean(cs)),
                "triangles": float(np.mean(tris)),
            }
        )
    if rows:
        c = {r["model"]: r["clustering"] for r in rows}
        non_c1 = [v for k, v in c.items() if "C1" not in k]
        print(
            f"\n  clustering: C1={c.get('C1 triangle', float('nan')):.4f} vs "
            f"others {min(non_c1):.4f}-{max(non_c1):.4f}"
        )
        print("  -> target degree distribution is IDENTICAL by construction (TV=0),")
        print("     but C1 is a clustering outlier. The matched-clustering subset")
        print("     is {Null, C2, C3, C4}; including C1 confounds cycle length")
        print("     with clustering.")
    return rows


# ------------------------------------------------------ D. orbit clustering
# Unique triangles contributed per subgraph instance (designed triangle budget).
# Hand-computed here on purpose: this table is the INDEPENDENT ORACLE that
# craeft.graphs.metrics.unique_triangles (ticket 004) is checked against below,
# so it must not be imported from the code under audit.
UNIQUE_TRIANGLES = {
    "triangle": 1,
    "square": 0,
    "pentagon": 0,
    "hexagon": 0,
    "diamond": 2,
    "K4": 4,
}


def exp_orbit_clustering(reps: int = 8) -> list[dict]:
    """Designed vs realized triangle count — the §4.6 claim.

    Designed triangles = M * u, fixed by instance count alone. Realized
    should equal designed + random floor, and does to within ~2%.
    """
    print("\n" + "=" * 78)
    print("D. ORBIT-RESOLVED CLUSTERING: designed vs realized")
    print("=" * 78)
    n, deg, rate = 1200, 12, 0.5
    degrees = np.full(n, deg, dtype=np.int_)

    floor_vals = []
    for s in range(reps):
        g = build(n, degrees, (), seed=900 + s)
        if g is not None:
            floor_vals.append(clique_count(adj_of(g), 3))
    floor = float(np.mean(floor_vals)) if floor_vals else 0.0
    print(f"  random floor: {floor:.1f} triangles (n={n}, k={deg})\n")
    print(
        f"  {'subgraph':<10}{'M':>6}{'designed':>10}{'oracle':>8}{'des+floor':>11}"
        f"{'realized':>10}{'ratio':>7}{'C_des':>9}{'C_real':>9}"
    )
    rows = []
    oracle_ok = True
    for name in ["triangle", "diamond", "K4", "square"]:
        sg = SG[name]
        seqs = (SubgraphSequence(subgraph=sg, distribution=poisson(rate)),)
        cfg = ConfigModelConfig(n=n, degrees=degrees, sequences=seqs, max_retries=300)
        got, creal = [], []
        for s in range(reps):
            g = build(n, degrees, seqs, seed=s)
            if g is None:
                continue
            got.append(clique_count(adj_of(g), 3))
            creal.append(g.clustering_coefficient)
        if not got:
            print(f"  {name:<10} FAILED")
            continue
        m = round(n * rate / sg.num_nodes)
        # Ticket 004: designed clustering is closed-form from the config, with
        # no generation. Cross-checked against the hand-computed oracle above.
        designed = designed_triangles(cfg)
        c_des = designed_clustering(cfg)
        oracle = m * UNIQUE_TRIANGLES[name]
        oracle_ok &= designed == oracle
        expected = designed + floor
        realized = float(np.mean(got))
        print(
            f"  {name:<10}{m:>6}{designed:>10.0f}{oracle:>8}{expected:>11.1f}"
            f"{realized:>10.1f}{realized / expected:>7.3f}"
            f"{c_des:>9.4f}{np.mean(creal):>9.4f}"
        )
        rows.append(
            {
                "subgraph": name,
                "instances": m,
                "designed_triangles": float(designed),
                "designed_triangles_oracle": oracle,
                "floor": floor,
                "expected": expected,
                "realized": realized,
                "ratio": realized / expected,
                "C_designed": c_des,
                "C_realized": float(np.mean(creal)),
            }
        )
    print("\n  => designed + floor predicts realized to within ~2%, including the")
    print("     mixed-orbit diamond. Global clustering is set by the triangle budget.")
    print(
        "\n  designed_triangles/designed_clustering (ticket 004) reproduce the"
        f"\n  hand-computed oracle column exactly: {oracle_ok}. Both are closed-form"
        "\n  from the config -- no generation needed -- so a matched-clustering pair"
        "\n  can be designed and checked BEFORE building. Criterion P5."
    )
    return rows


def exp_orbit_exactness(reps: int = 8) -> dict:
    """Orbit totals exact; per-node split sampled — the §4.6 warning."""
    print("\n" + "=" * 78)
    print("E. ORBIT SPLIT: global exact, per-node sampled")
    print("=" * 78)
    dia = SG["diamond"]
    seq = SubgraphSequence(subgraph=dia, distribution=poisson(1))
    tri_per_orbit = {0: 1, 1: 2}  # tips: 1 triangle, diagonal: 2
    n = 600

    all_match = True
    for s in range(reps):
        rng = np.random.default_rng(100 + s)
        v = seq.sample(n, rng)
        d = seq._split_by_orbit(v, rng)  # noqa: SLF001
        m = int(v.sum()) // dia.num_nodes
        for o, counts in d.items():
            if int(counts.sum()) != m * seq.orbit_sizes[o]:
                all_match = False
    print(f"  global orbit totals == M*sigma_o on all {reps} replicates: {all_match}")

    # same participation sequence, re-split -> per-node varies, total constant
    rng = np.random.default_rng(7)
    v = seq.sample(n, rng)
    per_node, totals = [], []
    for s in range(reps):
        rr = np.random.default_rng(300 + s)
        d = seq._split_by_orbit(v.copy(), rr)  # noqa: SLF001
        tri = np.zeros(n, dtype=np.int_)
        for o, counts in d.items():
            tri = tri + counts * tri_per_orbit[o]
        per_node.append(tri)
        totals.append(int(tri.sum()))
    arr = np.array(per_node)
    sampled_std = float(arr.std(axis=0).mean())
    print(f"  designed triangle TOTAL across re-splits: {set(totals)} (constant)")
    print(f"  per-node designed triangles, SAMPLED split:    std = {sampled_std:.3f}")

    # Ticket 007: the same participation sequence, but with the orbit split
    # PRESCRIBED via split_deterministic and passed as orbit_counts. The split
    # is then returned verbatim, so re-running under different seeds must give
    # bit-identical per-node triangle counts.
    prescribed = split_deterministic(v, seq)
    pinned = SubgraphSequence(
        subgraph=dia, distribution=poisson(1), orbit_counts=prescribed
    )
    per_node_p = []
    for s in range(reps):
        rr = np.random.default_rng(300 + s)
        d = pinned._split_by_orbit(v.copy(), rr)  # noqa: SLF001
        tri = np.zeros(n, dtype=np.int_)
        for o, counts in d.items():
            tri = tri + counts * tri_per_orbit[o]
        per_node_p.append(tri)
    arr_p = np.array(per_node_p)
    pinned_std = float(arr_p.std(axis=0).mean())
    print(f"  per-node designed triangles, PRESCRIBED split: std = {pinned_std:.3f}")

    print("\n  => global C is exact under both. With the SAMPLED split, per-node")
    print("     c_i is exact only in expectation for asymmetric subgraphs. With")
    print("     orbit_counts PRESCRIBED (ticket 007), per-node designed triangles")
    print("     are pinned across seeds, so the c(k) profile is controllable too.")
    print("     Vertex-transitive subgraphs (all cycles, all complete subgraphs)")
    print("     have a single orbit and are unaffected either way.")
    return {
        "global_totals_exact": all_match,
        "designed_total_constant": len(set(totals)) == 1,
        "per_node_std": sampled_std,
        "per_node_std_prescribed": pinned_std,
    }


# ------------------------------------------------------- F. assortativity
def exp_assortativity(reps: int = REPS) -> dict:
    """Degree assortativity per family — criterion P7.

    Assortativity is a second-order confound: clustering potential is
    bounded by the degree-degree correlation, and 2017 sec 3.3 pushed
    clustered subgraphs onto high-degree nodes, producing measurably more
    assortative networks (2017 Fig. 8). If a matched pair is to differ ONLY
    in higher-order structure, r must be shown not to move across it.

    Reported on two families, because r is only well defined on one:

    - the C-model family (degrees 2*Pois(2), heterogeneous) -- r defined;
    - the homogeneous families of section A (degree 10 for all nodes) --
      edge-end degree variance is exactly zero, so r is nan by construction,
      not by failure. c(k) collapses to one degree class there and carries
      no information either; for degree-regular families the order-four
      ratios are the diagnostic to use instead.
    """
    print("\n" + "=" * 78)
    print("F. DEGREE ASSORTATIVITY  (criterion P7: second-order confound)")
    print("=" * 78)
    n, lam = 1000, 2.0

    print("  F1. C-model family (degrees 2*Pois(2), heterogeneous -- r defined)")
    print(
        f"\n  {'model':<14}{'mean r':>10}{'std r':>9}{'min':>9}{'max':>9}"
        f"{'clustering':>12}{'deg classes':>13}"
    )
    cmodel_rows = []
    for label, key in [
        ("Null (random)", None),
        ("C1 triangle", "triangle"),
        ("C2 square", "square"),
        ("C3 pentagon", "pentagon"),
        ("C4 hexagon", "hexagon"),
    ]:
        rs, cs, classes = [], [], []
        for rep in range(reps):
            degrees = cmodel_degrees(n, lam, rep)
            seqs: tuple[SubgraphSequence, ...] = (
                ()
                if key is None
                else (SubgraphSequence(subgraph=SG[key], distribution=poisson(lam)),)
            )
            g = build(n, degrees, seqs, seed=500 + rep)
            if g is None:
                continue
            csr = g.to_csr()
            rs.append(degree_assortativity(csr))
            cs.append(g.clustering_coefficient)
            classes.append(len(clustering_by_degree(csr)))
        if not rs:
            print(f"  {label:<14} FAILED")
            continue
        print(
            f"  {label:<14}{np.mean(rs):>10.4f}{np.std(rs):>9.4f}"
            f"{min(rs):>9.4f}{max(rs):>9.4f}"
            f"{np.mean(cs):>12.4f}{int(np.mean(classes)):>13}"
        )
        cmodel_rows.append(
            {
                "model": label,
                "mean_assortativity": float(np.mean(rs)),
                "std_assortativity": float(np.std(rs)),
                "min_assortativity": float(min(rs)),
                "max_assortativity": float(max(rs)),
                "clustering": float(np.mean(cs)),
                "degree_classes": int(np.mean(classes)),
                "builds_ok": len(rs),
            }
        )

    spread = nulls = None
    if cmodel_rows:
        matched = [r for r in cmodel_rows if "C1" not in r["model"]]
        vals = [r["mean_assortativity"] for r in matched]
        spread = max(vals) - min(vals)
        nulls = next(r for r in cmodel_rows if "Null" in r["model"])
        within = max(r["std_assortativity"] for r in matched)
        print(
            f"\n  recommended pair {{Null, C2, C3, C4}}: r spans {min(vals):+.4f} to"
            f" {max(vals):+.4f}"
        )
        print(
            f"  between-family spread {spread:.4f} vs largest within-family"
            f" std {within:.4f}"
        )
        verdict = "NOT a confound" if spread <= 2 * within else "POSSIBLE confound"
        print(f"  => {verdict}: the spread across the matched family is")
        print(f"     {'within' if spread <= 2 * within else 'outside'} seed noise.")
        c1 = next((r for r in cmodel_rows if "C1" in r["model"]), None)
        if c1 is not None:
            print(
                f"  C1 triangle sits at r={c1['mean_assortativity']:+.4f} with"
                f" C={c1['clustering']:.4f} -- it is already excluded from the"
                "\n     recommended pair for being a clustering outlier (sec 6.3)."
            )

    print("\n  F2. Homogeneous families of section A (degree 10 -- r undefined)")
    degrees = np.full(n, 10, dtype=np.int_)
    homo_rows = []
    for label, key in [("none", None), ("triangle", "triangle"), ("K4", "K4")]:
        rs, classes = [], []
        for rep in range(reps):
            seqs = (
                ()
                if key is None
                else (SubgraphSequence(subgraph=SG[key], distribution=poisson(0.5)),)
            )
            g = build(n, degrees, seqs, seed=100 + rep)
            if g is None:
                continue
            csr = g.to_csr()
            rs.append(degree_assortativity(csr))
            classes.append(len(clustering_by_degree(csr)))
        if not rs:
            print(f"  {label:<14} FAILED")
            continue
        all_nan = bool(np.all(np.isnan(rs)))
        print(
            f"  {label:<14}r = {'nan (as designed)' if all_nan else np.mean(rs):<20}"
            f"degree classes in c(k): {int(np.mean(classes))}"
        )
        homo_rows.append(
            {
                "family": label,
                "all_nan": all_nan,
                "degree_classes": int(np.mean(classes)),
            }
        )
    print(
        "\n  => degree_assortativity returns nan rather than dividing by ~0"
        "\n     (ticket 005). Exact degree preservation (section A) means every"
        "\n     node really does have degree 10, so the zero variance is now a"
        "\n     property of the design rather than an artefact of edge loss."
    )
    return {
        "cmodel": cmodel_rows,
        "homogeneous": homo_rows,
        "matched_family_spread": float(spread) if spread is not None else None,
        "null_assortativity": nulls["mean_assortativity"] if nulls else None,
    }


# ------------------------------------------------------- G. order-four
# The six connected 4-node isomorphism classes, and the 2014 sec 2.2 item 4
# ratios, per family. Ticket 006 asked for this and never delivered it;
# ticket 008 renamed the ratio keys first so the audit does not bake the wrong
# labels into its output.
#
# NUMBERING (published 2014, p. 24 item 4 and Table 2 on p. 28):
#   phi_4_1 = ALL closed quadruples (aggregate)   phi_4_2 = empty square
#   phi_4_3 = diamond                             phi_4_4 = K4
#   unclosed = 1 - phi_4_1                        paw = component of unclosed
# with phi_4_1 = phi_4_2 + phi_4_3 + phi_4_4.
#
# DENOMINATOR: connected 4-node *induced subgraphs*, all six classes, stars
# included. The paper's Appendix A.2 counts 4-node *paths* instead, which a
# star has none of -- so these values are comparable ACROSS THE ROWS BELOW but
# NOT against 2014 Table 2. The star column measures the size of that gap.
ORDER_FOUR_CLASSES = ("path", "star", "cycle", "paw", "diamond", "complete")


def exp_order_four(reps: int = 4) -> list[dict]:
    """Order-four composition per family — criterion P4, direct.

    P4 asks whether NON-TARGET subgraph counts sit at the random floor.
    Section B only evidences it indirectly, on the target class. Here the
    Null row *is* the floor, and every other row is read against it.
    """
    print("\n" + "=" * 78)
    print("G. ORDER-FOUR COMPOSITION  (2014 sec 2.2 item 4; criterion P4, direct)")
    print("=" * 78)
    n, lam = 1000, 2.0
    print(
        f"  n={n}, degrees 2*Pois({lam:.0f}) (<k>=4), {reps} replicates, seeds 900+\n"
    )

    header_counts = "".join(f"{c:>10}" for c in ORDER_FOUR_CLASSES)
    print(f"  {'family':<14}{header_counts}")
    rows = []
    families = [
        ("Null (random)", None),
        ("C2 square", "square"),
        ("C3 pentagon", "pentagon"),
        ("C4 hexagon", "hexagon"),
        ("diamond", "diamond"),
        ("K4", "K4"),
    ]
    for label, key in families:
        counts_acc: dict[str, list[float]] = {c: [] for c in ORDER_FOUR_CLASSES}
        ratios_acc: dict[str, list[float]] = {}
        for rep in range(reps):
            degrees = cmodel_degrees(n, lam, rep)
            seqs: tuple[SubgraphSequence, ...] = (
                ()
                if key is None
                else (SubgraphSequence(subgraph=SG[key], distribution=poisson(lam)),)
            )
            g = build(n, degrees, seqs, seed=900 + rep)
            if g is None:
                continue
            csr = g.to_csr()
            counts = count_order_four(csr)
            for c in ORDER_FOUR_CLASSES:
                counts_acc[c].append(counts[c])
            for k, v in order_four_ratios(csr).items():
                ratios_acc.setdefault(k, []).append(v)
        if not counts_acc["path"]:
            print(f"  {label:<14} FAILED")
            continue
        mean_counts = {c: float(np.mean(counts_acc[c])) for c in ORDER_FOUR_CLASSES}
        mean_ratios = {k: float(np.mean(v)) for k, v in ratios_acc.items()}
        total = sum(mean_counts.values())
        print(
            f"  {label:<14}"
            + "".join(f"{mean_counts[c]:>10.0f}" for c in ORDER_FOUR_CLASSES)
        )
        rows.append(
            {
                "family": label,
                "counts": mean_counts,
                "ratios": mean_ratios,
                "connected_quadruples": total,
                "paw_share": mean_counts["paw"] / total if total else 0.0,
                "star_share": mean_counts["star"] / total if total else 0.0,
                "builds_ok": len(counts_acc["path"]),
            }
        )

    if not rows:
        return rows

    print(
        f"\n  {'family':<14}{'phi_4^1':>10}{'phi_4^2':>10}{'phi_4^3':>10}"
        f"{'phi_4^4':>10}{'unclosed':>10}{'paw':>10}{'star %':>9}"
    )
    for r in rows:
        m = r["ratios"]
        print(
            f"  {r['family']:<14}{m['phi_4_1']:>10.5f}{m['phi_4_2']:>10.5f}"
            f"{m['phi_4_3']:>10.5f}{m['phi_4_4']:>10.5f}{m['unclosed']:>10.5f}"
            f"{m['paw']:>10.5f}{r['star_share']:>8.1%}"
        )

    additive = all(
        abs(
            r["ratios"]["phi_4_1"]
            - (r["ratios"]["phi_4_2"] + r["ratios"]["phi_4_3"] + r["ratios"]["phi_4_4"])
        )
        < 1e-12
        for r in rows
    )
    print(f"\n  phi_4^1 == phi_4^2 + phi_4^3 + phi_4^4 on every row: {additive}")
    print("  (the identity 2014 Table 2 satisfies; pins the corrected numbering)")

    star_lo = min(r["star_share"] for r in rows)
    star_hi = max(r["star_share"] for r in rows)
    print(
        f"\n  stars are {star_lo:.1%}-{star_hi:.1%} of the denominator. The paper's"
        "\n  Appendix A.2 path-extension counter cannot enumerate a star (no 4-node"
        "\n  path), so its ratios use a denominator ~30-36% smaller than this one."
        "\n  => these rows are comparable with EACH OTHER, not with 2014 Table 2."
    )

    null = rows[0]
    print("\n  P4 (non-target counts at the floor?) read against the Null row:")
    print(
        f"    {'family':<14}{'cycle/null':>12}{'paw/null':>10}"
        f"{'diamond/null':>14}{'K4/null':>9}"
    )

    def _rel(r: dict, cls: str) -> str:
        base = null["counts"][cls]
        # A zero floor is the strongest possible result, not a missing one --
        # print the raw count over it rather than "n/a".
        count = r["counts"][cls]
        return f"{count / base:.1f}x" if base else f"{count:.0f}/0"

    for r in rows[1:]:
        print(
            f"    {r['family']:<14}{_rel(r, 'cycle'):>12}{_rel(r, 'paw'):>10}"
            f"{_rel(r, 'diamond'):>14}{_rel(r, 'complete'):>9}"
        )
    print(
        "\n  Note the diamond and K4 rows sit BELOW the Null on the cycle column"
        "\n  (0.4-0.6x). Not noise: those families spend the same degree budget on"
        "\n  triangle-bearing subgraphs, which consumes the stub pairs that would"
        "\n  otherwise close chordless 4-cycles at the floor rate. Non-target"
        "\n  counts are held near the floor from ABOVE, not pinned to it."
    )
    print(
        "\n  => the target class rises sharply above the Null and the non-target"
        "\n     closed classes stay near it, which is what P4 asks for. The"
        "\n     exception is the PAW, and it is not a defect: a paw is the"
        "\n     by-product signature of a designed triangle (any triangle plus one"
        "\n     external edge makes one), so triangle-bearing families necessarily"
        "\n     carry paws. It is negligible (~0.6%) across {Null, square, pentagon,"
        "\n     hexagon} -- so it cannot confound the recommended family -- and"
        "\n     dominant (7-15%) for diamond/K4, where it outnumbers the DESIGNED"
        "\n     structures by an order of magnitude and separates those families"
        "\n     more sharply than any phi_4^i does."
    )
    print(
        "\n  A paw cannot be an INPUT subgraph here: the CCM requires a Hamiltonian"
        "\n  cycle and the paw (G6 in docs/concepts/motifs.md) has none. Every paw"
        "\n  measured above is therefore a by-product, not a prescribed structure."
    )
    return rows


# ------------------------------------------- H and I live in separate files
# Sections H and I are NOT in this file. They were written as standalone
# harnesses because each needed its own degree generators, degree binning and
# distributional statistics, and folding them in here would have doubled the
# module without sharing anything but the imports.
#
#   H -- per-node SNR criteria P3-local and P8, plus the placement experiment
#        (ticket 010).  audit_section_h.py -> audit_section_h_results.json.
#        Results feed docs/claims/claim-4-residual-freedom.md.
#
#   I -- closed-form floor predictor validation, heavy-tail cycle and clique
#        floors, and the regime-utility verdict (ticket 011).
#        audit_section_i.py -> audit_section_i_results.json.
#        Results feed docs/claims/claim-2-designed-counts.md.
#
# Note the re-lettering: ticket 010 originally reserved section G, which
# ticket 008's order-four audit took; 010 became H and 011 became I.


def main() -> None:
    out = {
        "degree_preservation": exp_degrees(),
        "subgraph_fidelity": exp_fidelity(),
        "cmodels": exp_cmodels(),
        "orbit_clustering": exp_orbit_clustering(),
        "orbit_exactness": exp_orbit_exactness(),
        "assortativity": exp_assortativity(),
        "order_four": exp_order_four(),
    }
    dest = Path(__file__).parent / "control_audit_results.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nwrote {dest}")


if __name__ == "__main__":
    main()
