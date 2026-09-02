"""Sampler-uniformity audit for the matching (collision-reselection) pairing.

Quantifies, exactly, how far ``Connector.connect_singles`` deviates from
uniform sampling over simple graphs with a fixed degree sequence. Evidence
for docs/claims/claim-1-exact-degrees.md ("the cost, quantified").

Three pillars, each checking the next:

  1. Exact distribution (rational arithmetic). The sampler reduces to a
     Markov chain whose state is the set of committed edges: the anchor
     stub is uniform (last element of a uniformly shuffled list), the
     partner is uniform over the rest, and a collision reshuffles without
     changing the state — a pure self-transition. Conditional on leaving
     a state, edge {u, v} is committed with probability s_u s_v / W,
     where s are residual stub counts and W sums s_a s_b over valid
     pairs (distinct nodes, edge not present). A dynamic programme over
     edge sets gives every graph's probability as an exact fraction;
     dead-end mass (non-graphical residuals, the ConnectionError ->
     restart-afresh path in from_config) is renormalised out exactly as
     production does.

  2. Closed form for P(g). The claim doc derives, edge by edge, the
     exact expression

         P_raw(g) = (prod_u d_u!) * sum_{orderings pi of E(g)}
                        prod_t 1 / W_t(pi):

     node u's residual count takes each value d_u, ..., 1 exactly once
     across its edges, so the numerator telescopes to prod d_u! for
     every ordering. Corollary: were W_t graph-independent (restart-from-
     scratch rejection), P(g) would be constant — the classical uniformity
     of rejection sampling. All bias therefore lives in W_t's two
     collision-exclusion terms. The script evaluates the closed form for
     every graph and checks it twice. Against the dynamic programme of
     pillar 1; and from outside the derivation: with both exclusion terms
     removed, W_t = T_t (T_t - 1) / 2 depends on the step alone, the sum
     over orderings collapses to 1/(l_n - 1)!!, and the closed form must
     return the configuration model's own law for a simple graph,

         P(CM_n(d) = g) = prod_u d_u! / (l_n - 1)!!

     (van der Hofstad, Random Graphs and Complex Networks, Vol. 1, Ch. 7,
     eq. (7.5.2) via Proposition 7.7; lecture-notes edition at
     https://rhofstad.win.tue.nl/Cap_Sel_Connectivity_in_RG.html). The
     book reaches the same prod d_u! by an unrelated static count — the
     half-edge permutations at each vertex that leave the graph unchanged
     — so the telescoping step is checked against an independent
     derivation, not only against pillar 1. Summed over the simple graphs
     the reference gives P(CM_n(d) simple), hence the restart rate of
     restart-from-scratch rejection sampling (the book's "repeated
     configuration model"): the reference point for the dead-end mass of
     pillar 1. A first-order expansion of 1/W_t around the same
     mean-field value A_t = T_t (T_t - 1) / 2 predicts the deviation as
     B(g) - mean(B), with B(g) the ordering-averaged sum of excluded
     collision weight over A_t; the prediction is reported next to the
     exact value (right sign, understated magnitude — higher-order terms
     matter at this size, where dead-end mass is ~31%; the symmetric +/-
     split is forced by the sequence's two equal orbits, not predicted).

  3. Monte Carlo of the real implementation. Repeated builds through
     Connector.connect_singles, restarting on ConnectionError as
     from_config does. Validates the step the reduction cannot get for
     free: after a commit without reshuffle, the remaining stub order is
     still uniform (an exchangeability lemma). Agreement is reported as
     z-scores of observed frequencies against the exact probabilities.

  4. First-order structure and the size limit. Write W_t = A_t - X_t.
     Then P_raw(g) = [prod d_u! / (l_n - 1)!!] * R(g) with R(g) the
     ordering average of prod_t 1/(1 - X_t/A_t), and log R(g) = B(g) to
     first order. B splits into a self-loop part and a duplicate part.
     The self-loop part is exactly graph-independent (when node u's
     edges are committed has the same law for every graph with the same
     degrees). The duplicate part is exactly Phi_m * S(g), where
     S(g) = sum over edges of (d_a - 1)(d_b - 1) and Phi_m ~ 1/(4m) has
     the closed form in phi_coefficient. So to first order the sampler
     over-weights graphs by exp(Phi_m S(g)): an excess-degree
     assortativity bias, vanishing identically for regular sequences.
     The size scan tiles the audit sequence k times, draws exactly
     uniform simple graphs by rejection, and reports the first-order
     spread Phi_m * std(S): it falls like 1/sqrt(m) while the worst
     case Phi_m * (max S - min S) stays O(1). Literature: for regular
     sequences this sampler is asymptotically uniform (Steger & Wormald,
     Combin. Probab. Comput. 8 (1999) 377-396; Kim & Vu, STOC 2003,
     d = O(n^{1/3-eps})); for irregular ones Bayati, Kim & Saberi
     (Algorithmica 58 (2010) 860-910) cancel this first-order term with
     a (1 - d_a d_b / 4m) pair weight and prove near-uniformity for
     d_max = O(m^{1/4-eps}).

The audit degree sequence (3, 2, 2, 2, 1) is small enough to enumerate
(six simple graphs) and heterogeneous enough to show the mechanism: the
three under-sampled graphs are exactly those containing the hub-leaf
edge (0, 4).

Run:
  .venv/bin/python docs/claims/evidence/uniformity_bias.py
    -> docs/claims/evidence/uniformity_bias_results.json
"""

from __future__ import annotations

import json
from collections import defaultdict
from fractions import Fraction
from itertools import combinations, permutations
from math import factorial, prod, sqrt
from pathlib import Path

import numpy as np

from craeft.graphs.configuration_model.connection import (
    ConnectionError,
    Connector,
)

DEGREES = (3, 2, 2, 2, 1)
N = len(DEGREES)
M = sum(DEGREES) // 2
STUBS = sum(DEGREES)  # l_n, the number of half-edges
STUB_FACTOR = prod(factorial(d) for d in DEGREES)  # prod_u d_u!


def double_factorial(n: int) -> int:
    return 1 if n <= 0 else n * double_factorial(n - 2)


# The configuration model's law for a simple graph: the probability that
# uniform half-edge pairing produces a given simple graph with degrees
# DEGREES. van der Hofstad, Random Graphs and Complex Networks, Vol. 1,
# eq. (7.5.2) via Proposition 7.7.
CM_REFERENCE = Fraction(STUB_FACTOR, double_factorial(STUBS - 1))
CM_SOURCE = (
    "van der Hofstad, Random Graphs and Complex Networks, Vol. 1, Ch. 7, "
    "eq. (7.5.2) via Proposition 7.7"
)
CM_SOURCE_URL = "https://rhofstad.win.tue.nl/Cap_Sel_Connectivity_in_RG.html"

MC_SAMPLES = 200_000
MC_SEED = 20260830


Graph = frozenset  # frozenset of (u, v) tuples, u < v


def edge_label(g: Graph) -> str:
    return ",".join(f"{u}{v}" for u, v in sorted(g))


def residuals(edges) -> list[int]:
    s = list(DEGREES)
    for u, v in edges:
        s[u] -= 1
        s[v] -= 1
    return s


def pair_weight(edges) -> Fraction:
    """Total valid-pair weight W from a state (set of committed edges).

    W = T(T-1)/2 - sum_u s_u(s_u-1)/2 - sum_{(a,b) in E} s_a s_b: every
    pair of distinct stubs, less the two exclusion terms (same-node pairs,
    which would self-loop, and pairs that would duplicate a committed
    edge). mean_field_weight is the first term alone.
    """
    s = residuals(edges)
    total = sum(s)
    all_pairs = Fraction(total * (total - 1), 2)
    self_loops = sum(Fraction(x * (x - 1), 2) for x in s)
    duplicates = sum(s[a] * s[b] for a, b in edges)
    return all_pairs - self_loops - duplicates


def mean_field_weight(edges) -> Fraction:
    """W with both exclusion terms removed: T (T - 1) / 2, every unordered
    stub pair admissible. Pairing with this weight is the uniform adaptable
    pairing of the configuration model itself (van der Hofstad, Definition
    7.5), so the closed form built on it must reproduce CM_REFERENCE."""
    total = sum(residuals(edges))
    return Fraction(total * (total - 1), 2)


# ------------------------------------------------- 1. exact distribution
def exact_distribution() -> tuple[dict[Graph, Fraction], Fraction]:
    """Exact raw graph probabilities and dead-end mass, as fractions."""
    level: dict[Graph, Fraction] = {frozenset(): Fraction(1)}
    dead = Fraction(0)
    for _ in range(M):
        nxt: dict[Graph, Fraction] = defaultdict(Fraction)
        for edges, p in level.items():
            s = residuals(edges)
            valid = [
                (u, v, s[u] * s[v])
                for u, v in combinations(range(N), 2)
                if s[u] > 0 and s[v] > 0 and (u, v) not in edges
            ]
            total = sum(w for *_, w in valid)
            if total == 0:
                dead += p
                continue
            for u, v, w in valid:
                nxt[edges | {(u, v)}] += p * Fraction(w, total)
        level = nxt
    return dict(level), dead


def enumerate_reference() -> list[Graph]:
    """All simple graphs with the audit degree sequence, by brute force."""
    graphs = []
    for subset in combinations(list(combinations(range(N), 2)), M):
        deg = [0] * N
        for u, v in subset:
            deg[u] += 1
            deg[v] += 1
        if tuple(deg) == DEGREES:
            graphs.append(frozenset(subset))
    return graphs


# ---------------------------------------- 2. closed form and first order
def ordering_sum(g: Graph, weight=pair_weight) -> Fraction:
    """P_raw(g) via the derived closed form: prod d_u! * sum_pi prod 1/W.

    ``weight`` maps a state to its normaliser W. With ``pair_weight`` this
    is the retrying sampler; with ``mean_field_weight`` the exclusions are
    gone and the result must equal CM_REFERENCE, eq. (7.5.2), for every
    simple graph.
    """
    phi = Fraction(0)
    for pi in permutations(sorted(g)):
        prob = Fraction(1)
        state: Graph = frozenset()
        for edge in pi:
            prob /= weight(state)
            state = state | {edge}
        phi += prob
    return STUB_FACTOR * phi


def first_order_B(g: Graph) -> Fraction:
    """Ordering-averaged sum of excluded collision weight over A_t."""
    acc = Fraction(0)
    count = 0
    for pi in permutations(sorted(g)):
        state: Graph = frozenset()
        for edge in pi:
            s = residuals(state)
            mean_field = mean_field_weight(state)
            excluded = Fraction(sum(x * (x - 1) for x in s), 2) + sum(
                s[a] * s[b] for a, b in state
            )
            acc += excluded / mean_field
            state = state | {edge}
        count += 1
    return acc / count


# ------------------------------- 4. first-order structure and the limit
def phi_coefficient(m: int) -> Fraction:
    """Phi_m: first-order duplicate weight per unit of excess-degree
    product. For an edge (a, b) at a uniformly random position among m,
    the ordering-averaged sum over later steps of s_a s_b / A_t equals
    (d_a - 1)(d_b - 1) Phi_m exactly, with

        Phi_m = sum_{j=1}^{m-2} j (m-1-j) / ((2j+1) m (m-1)(m-2))
              = [m(m-2) - (2m-1) H_m] / (4 m (m-1)(m-2)),

    H_m = sum_{j=1}^{m-2} 1/(2j+1); hence Phi_m ~ 1/(4m). The step
    behind it: given the edge is among the first t-1 commits, the
    counts X, Y of a's and b's other edges among the remaining t-2 are
    multivariate hypergeometric on m-1 slots, so with p = d_a - 1 and
    q = d_b - 1, E[(p-X)(q-Y)] = pq (m-t+1)(m-t) / ((m-1)(m-2)).
    """
    h = sum(Fraction(1, 2 * j + 1) for j in range(1, m - 1))
    return (m * (m - 2) - (2 * m - 1) * h) / (4 * m * (m - 1) * (m - 2))


def phi_coefficient_sum(m: int) -> Fraction:
    """Phi_m as the bare sum, to check the closed form."""
    return sum(
        Fraction(j * (m - 1 - j), (2 * j + 1) * m * (m - 1) * (m - 2))
        for j in range(1, m - 1)
    )


def excess_product_sum(edges, degrees) -> int:
    """S(g): sum over edges of (d_a - 1)(d_b - 1)."""
    return sum((degrees[a] - 1) * (degrees[b] - 1) for a, b in edges)


def first_order_split(g: Graph) -> tuple[Fraction, Fraction]:
    """first_order_B split into its self-loop and duplicate parts."""
    loops = Fraction(0)
    dups = Fraction(0)
    count = 0
    for pi in permutations(sorted(g)):
        state: Graph = frozenset()
        for edge in pi:
            s = residuals(state)
            mean_field = mean_field_weight(state)
            loops += Fraction(sum(x * (x - 1) for x in s), 2) / mean_field
            dups += sum(s[a] * s[b] for a, b in state) / mean_field
            state = state | {edge}
        count += 1
    return loops / count, dups / count


SCAN_TILES = (1, 2, 4, 8, 16, 32, 64)
SCAN_SAMPLES = 400
SCAN_SEED = 20260902


def sample_uniform_simple(degrees, rng, n_samples):
    """Exactly uniform simple graphs by the repeated configuration
    model: pair stubs uniformly, reject unless simple."""
    n = len(degrees)
    stubs = np.repeat(np.arange(n), degrees)
    out = []
    attempts = 0
    while len(out) < n_samples:
        attempts += 1
        perm = rng.permutation(stubs)
        u, v = perm[0::2], perm[1::2]
        if np.any(u == v):
            continue
        lo, hi = np.minimum(u, v), np.maximum(u, v)
        if len(np.unique(lo * n + hi)) < len(lo):
            continue
        out.append(list(zip(lo.tolist(), hi.tolist())))
    return out, attempts


def size_scan() -> list[dict]:
    """First-order bias spread across uniform graphs as m grows."""
    rng = np.random.default_rng(SCAN_SEED)
    rows = []
    for k in SCAN_TILES:
        degrees = DEGREES * k
        m = sum(degrees) // 2
        graphs, attempts = sample_uniform_simple(degrees, rng, SCAN_SAMPLES)
        s_vals = np.array(
            [excess_product_sum(g, degrees) for g in graphs], dtype=float
        )
        phi = float(phi_coefficient(m))
        dev = phi * (s_vals - s_vals.mean())
        rows.append(
            {
                "tiles": k,
                "nodes": len(degrees),
                "edges": m,
                "uniform_samples": SCAN_SAMPLES,
                "acceptance_rate": SCAN_SAMPLES / attempts,
                "phi_m": phi,
                "four_m_phi_m": 4 * m * phi,
                "S_mean": float(s_vals.mean()),
                "S_std": float(s_vals.std()),
                "typical_relative_bias": float(dev.std()),
                "max_relative_bias": float(np.abs(dev).max()),
                "first_order_tvd": float(0.5 * np.abs(dev).mean()),
                "sqrt_m_times_typical": float(sqrt(m) * dev.std()),
            }
        )
    return rows


# --------------------------------------- 3. Monte Carlo of the real code
def sample_implementation() -> tuple[dict[Graph, int], int]:
    rng = np.random.default_rng(MC_SEED)
    singles = np.array(DEGREES)
    counts: dict[Graph, int] = defaultdict(int)
    restarts = 0
    for _ in range(MC_SAMPLES):
        while True:
            conn = Connector(N, rng)
            try:
                conn.connect_singles(singles)
                break
            except ConnectionError:
                restarts += 1
        g = frozenset(
            (min(r, c), max(r, c)) for r, c in zip(conn._rows, conn._cols)
        )
        counts[g] += 1
    return counts, restarts


def main() -> None:
    raw, dead = exact_distribution()
    graphs = enumerate_reference()
    assert set(raw) == set(graphs), "DP support != enumerated simple graphs"

    z_norm = 1 - dead
    uniform = Fraction(1, len(graphs))
    exact = {g: raw[g] / z_norm for g in graphs}
    deviations = {g: exact[g] / uniform - 1 for g in graphs}
    tvd = sum(abs(exact[g] - uniform) for g in graphs) / 2

    closed_form_ok = {g: ordering_sum(g) == raw[g] for g in graphs}
    assert all(closed_form_ok.values()), "closed-form check vs DP failed"

    cm_law_ok = {
        g: ordering_sum(g, mean_field_weight) == CM_REFERENCE for g in graphs
    }
    assert all(cm_law_ok.values()), "mean-field closed form != eq. (7.5.2)"
    cm_simple = CM_REFERENCE * len(graphs)  # P(CM_n(d) is simple)
    repeated_restart = 1 - cm_simple  # rejection rate of the repeated CM

    b_values = {g: first_order_B(g) for g in graphs}
    b_mean = sum(b_values.values()) / len(graphs)

    # 4. first-order structure: exact split, closed form, size scan
    phi = phi_coefficient(M)
    assert phi == phi_coefficient_sum(M), "Phi_m closed form != sum"
    splits = {g: first_order_split(g) for g in graphs}
    loop_terms = {splits[g][0] for g in graphs}
    assert len(loop_terms) == 1, "self-loop term depends on the graph"
    s_of = {g: excess_product_sum(g, DEGREES) for g in graphs}
    assert all(splits[g][1] == phi * s_of[g] for g in graphs), (
        "duplicate term != Phi_m * S(g)"
    )
    assert all(sum(splits[g]) == b_values[g] for g in graphs)
    scan = size_scan()

    counts, restarts = sample_implementation()
    mc = {}
    for g in graphs:
        p = float(exact[g])
        freq = counts.get(g, 0) / MC_SAMPLES
        se = sqrt(p * (1 - p) / MC_SAMPLES)
        mc[g] = {"frequency": freq, "z_vs_exact": (freq - p) / se}
    max_abs_z = max(abs(row["z_vs_exact"]) for row in mc.values())

    out = {
        "degree_sequence": list(DEGREES),
        "num_simple_graphs": len(graphs),
        "dead_end_mass": {"fraction": str(dead), "value": float(dead)},
        "configuration_model_reference": {
            "source": CM_SOURCE,
            "url": CM_SOURCE_URL,
            "probability_per_simple_graph": {
                "fraction": str(CM_REFERENCE),
                "value": float(CM_REFERENCE),
            },
            "probability_simple": {
                "fraction": str(cm_simple),
                "value": float(cm_simple),
            },
            "repeated_model_restart_rate": {
                "fraction": str(repeated_restart),
                "value": float(repeated_restart),
            },
            "restart_rate_ratio_repeated_over_retrying": float(
                repeated_restart / dead
            ),
        },
        "metrics": {
            "max_abs_relative_deviation": float(
                max(abs(d) for d in deviations.values())
            ),
            "mean_abs_relative_deviation": float(
                sum(abs(d) for d in deviations.values()) / len(graphs)
            ),
            "total_variation_distance": float(tvd),
        },
        "per_graph": {
            edge_label(g): {
                "exact_probability": {
                    "fraction": str(exact[g]),
                    "value": float(exact[g]),
                },
                "relative_deviation_vs_uniform": float(deviations[g]),
                "first_order_prediction": float(b_values[g] - b_mean),
                "closed_form_verified": closed_form_ok[g],
                "cm_law_verified": cm_law_ok[g],
                "mc_frequency": mc[g]["frequency"],
                "mc_z_vs_exact": mc[g]["z_vs_exact"],
            }
            for g in sorted(graphs, key=edge_label)
        },
        "first_order_structure": {
            "phi_m": {
                "fraction": str(phi),
                "value": float(phi),
                "closed_form_verified": True,
            },
            "self_loop_term": {
                "fraction": str(next(iter(loop_terms))),
                "graph_independent": True,
            },
            "per_graph": {
                edge_label(g): {
                    "S": s_of[g],
                    "duplicate_term": str(splits[g][1]),
                    "equals_phi_times_S": True,
                }
                for g in sorted(graphs, key=edge_label)
            },
        },
        "size_scan": {
            "seed": SCAN_SEED,
            "tiled_sequence": list(DEGREES),
            "rows": scan,
        },
        "monte_carlo": {
            "samples": MC_SAMPLES,
            "seed": MC_SEED,
            "dead_end_restarts": restarts,
            "max_abs_z_vs_exact": max_abs_z,
        },
    }

    dest = Path(__file__).parent / "uniformity_bias_results.json"
    dest.write_text(json.dumps(out, indent=2))

    print(f"simple graphs: {len(graphs)}; dead-end mass {float(dead):.4f}")
    print(
        "max |relative deviation| "
        f"{out['metrics']['max_abs_relative_deviation']:.4%}; "
        f"TVD {out['metrics']['total_variation_distance']:.4f}"
    )
    print(f"closed form verified against the DP on all {len(graphs)} graphs")
    print(
        f"closed form with exclusions removed == eq. (7.5.2) reference "
        f"{CM_REFERENCE} on all {len(graphs)} graphs; P(simple) "
        f"{float(cm_simple):.4f}, repeated-model restart "
        f"{float(repeated_restart):.4f} vs retrying {float(dead):.4f}"
    )
    print(
        f"first order: Phi_m = {phi} ({float(phi):.5f}); duplicate term == "
        f"Phi_m * S(g) and self-loop term graph-independent on all "
        f"{len(graphs)} graphs"
    )
    for row in scan:
        print(
            f"  m={row['edges']:4d}  4m*Phi={row['four_m_phi_m']:.3f}  "
            f"typical bias={row['typical_relative_bias']:.4f}  "
            f"max={row['max_relative_bias']:.4f}  "
            f"sqrt(m)*typical={row['sqrt_m_times_typical']:.3f}"
        )
    print(f"MC max |z| vs exact: {max_abs_z:.2f} ({MC_SAMPLES} samples)")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
