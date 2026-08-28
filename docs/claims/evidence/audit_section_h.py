"""Section H of the control audit: LOCAL (per-node) signal and matching.

Ticket 010. The acceptance criteria in dev/algorithmic-claim.md section 0 are
global totals (P3, P4, P5); an L-layer message-passing network reads L-hop
neighbourhoods, so the quantities it can actually see are per-node. This
section measures the two local criteria the ticket proposes:

  H1  P3-local -- per-node target-motif incidence against the
      DEGREE-CONDITIONAL floor, stratified by degree class. Reports the
      fraction of all nodes, and separately of *participating* nodes, whose
      designed incidence clears 5x their own degree class's floor.

  H2  P8 -- for each motif a matched pair holds fixed, the distributional
      distance (TV on binned per-node incidence, KS as a bin-free check)
      between the pair's per-node incidence profiles, read against the
      within-family replicate scatter. A pair "leaks" when the between-family
      distance is outside that scatter.

  H3  the placement experiment -- heterogeneous pentagon family, default
      (capped, hub-correlated) placement vs prescribed placements that avoid
      the hubs. Tests the ticket's hypothesis that local SNR at participating
      nodes clears 5x even though global P3 fails (the family sits at 0.8x
      the floor, section 6.3).

Lives in its own file rather than in control_audit.py so the two can be
developed in parallel; it imports the harness for the family definitions,
seeding conventions and build wrapper. Merge into control_audit.py later.

Run:
  MPLCONFIGDIR=$TMPDIR .venv/bin/python dev/audit_section_h.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import ks_2samp, poisson

sys.path.insert(0, str(Path(__file__).parent))

import control_audit as ca  # noqa: E402

from craeft.graphs.configuration_model import SubgraphSequence  # noqa: E402
from craeft.graphs.configuration_model.sequence import (  # noqa: E402
    split_deterministic,
)
from craeft.graphs.metrics import (  # noqa: E402
    cycles_per_node,
    order_four_per_node,
)

# --------------------------------------------------------------- parameters
N = 1000
LAM = 2.0  # participation rate AND the C-model's degree parameter
REGULAR_DEGREE = 4

REPS_H1 = 12  # null replicates == signal replicates, per family per target
REPS_H2 = 12  # matches section C/F, so H2 reads against the same builds
REPS_H3 = 12

# Degree stream is 20_000 + rep (control_audit.DEGREE_SEED_BASE). Build and
# sampling streams must stay far from it -- see the HAZARD note in
# control_audit.py: a build seed that collides with the degree seed makes the
# participation sequence exactly equal to the degree-derived cap.
H1_NULL_SEED_BASE = 31_000
H1_BUILD_SEED_BASE = 32_000
H1_PARTICIPATION_SEED_BASE = 33_000
H2_SEED_BASE = 500  # deliberate reuse: same builds as 6.3 / 6.4
H3_NULL_SEED_BASE = 34_000
H3_BUILD_SEED_BASE = 35_000
H3_PARTICIPATION_SEED_BASE = 36_000
H3_PLACEMENT_SEED_BASE = 37_000

SNR_BAR = 5.0  # P3's ">= 5x the floor" bar, applied per degree class
MIN_BIN_OBS = 200  # node-observations needed before a degree class stands alone

CYCLE_TARGETS = (("triangle", 3), ("square", 4), ("pentagon", 5), ("hexagon", 6))


# --------------------------------------------------------------- helpers
def degree_bins(degrees: np.ndarray, reps: int) -> list[tuple[int, int]]:
    """Pool ascending degree classes until each bin holds MIN_BIN_OBS.

    Returns inclusive (lo, hi) degree ranges. Stratification is load-bearing
    for H1 (the floor concentrates on hubs), but a class with a handful of
    nodes gives a floor estimate that is pure noise -- so sparse classes are
    pooled upward and the last bin absorbs the tail.
    """
    classes = sorted(int(k) for k in set(degrees.tolist()))
    counts = {c: int((degrees == c).sum()) * reps for c in classes}
    bins: list[tuple[int, int]] = []
    lo: int | None = None
    acc = 0
    for c in classes:
        if lo is None:
            lo = c
        acc += counts[c]
        if acc >= MIN_BIN_OBS:
            bins.append((lo, c))
            lo, acc = None, 0
    if lo is not None:  # tail below threshold: merge into the previous bin
        if bins:
            bins[-1] = (bins[-1][0], classes[-1])
        else:
            bins.append((lo, classes[-1]))
    return bins


def bin_label(lo: int, hi: int, top: int) -> str:
    if lo == hi:
        return f"k={lo}"
    if hi >= top:
        return f"k>={lo}"
    return f"k={lo}-{hi}"


def prescribed_cycle_sequence(
    subgraph_key: str, participation: np.ndarray
) -> SubgraphSequence:
    """A sequence whose per-node placement is pinned, not sampled.

    Cycles are vertex-transitive, so split_deterministic is the identity on
    the orbit split; the point of routing through it is that the returned dict
    is what orbit_counts wants, and the construction generalises to
    asymmetric subgraphs unchanged. Prescribing rather than sampling is what
    makes the DESIGNED side of H1 exact: per-node designed incidence *is*
    the participation sequence, with no need to replay the builder's rng.
    """
    template = SubgraphSequence(subgraph=ca.SG[subgraph_key], distribution=poisson(LAM))
    return SubgraphSequence(
        subgraph=ca.SG[subgraph_key],
        orbit_counts=split_deterministic(participation.astype(np.int_), template),
    )


def capped_participation(
    subgraph_key: str, degrees: np.ndarray, seed: int
) -> np.ndarray:
    """The generator's own default draw: Pois(LAM) clipped to the degree cap.

    Reproduces exactly what ConfigModelGraph.from_config does on its first
    attempt -- max_per_node = degrees // max(orbit_degrees) -- so this arm is
    the *default* placement, hub-correlated by ticket 003's cap.
    """
    seq = SubgraphSequence(subgraph=ca.SG[subgraph_key], distribution=poisson(LAM))
    cap = degrees // max(seq.orbit_degrees.values())
    return seq.sample(N, np.random.default_rng(seed), max_per_node=cap)


def water_fill_placement(
    total: int, degrees: np.ndarray, cardinality: int, seed: int
) -> np.ndarray:
    """Degree-neutral placement: fill level by level, uniformly at random.

    Every node with slack under its cap is equally likely to receive the next
    participation, so within a fill level placement is blind to degree. The
    residual degree dependence is exactly the cap -- a node cannot hold more
    participations than its degree budget allows -- which no feasible
    placement can escape.
    """
    cap = degrees // cardinality
    placement = np.zeros(len(degrees), dtype=np.int_)
    rng = np.random.default_rng(seed)
    remaining = total
    while remaining > 0:
        avail = np.flatnonzero(placement < cap)
        if avail.size == 0:
            break
        rng.shuffle(avail)
        take = min(remaining, avail.size)
        placement[avail[:take]] += 1
        remaining -= take
    return placement


def low_degree_first_placement(
    total: int, degrees: np.ndarray, cardinality: int, seed: int
) -> np.ndarray:
    """Maximally hub-avoiding placement: saturate the periphery first.

    Not "degree-neutral" -- deliberately degree-ANTI-correlated. It is the
    best case any placement rule can achieve for local SNR, because the floor
    is monotone in degree, so it bounds what H3's hypothesis could possibly
    buy. Ties within a degree class are broken at random.
    """
    cap = degrees // cardinality
    placement = np.zeros(len(degrees), dtype=np.int_)
    rng = np.random.default_rng(seed)
    jitter = rng.random(len(degrees))
    order = np.lexsort((jitter, degrees))
    remaining = total
    for idx in order:
        if remaining <= 0:
            break
        take = min(int(cap[idx]), remaining)
        placement[idx] = take
        remaining -= take
    return placement


def total_variation(a: np.ndarray, b: np.ndarray, edges: np.ndarray) -> float:
    """TV distance between two per-node incidence profiles on shared bins."""
    pa, _ = np.histogram(a, bins=edges)
    pb, _ = np.histogram(b, bins=edges)
    pa = pa / max(pa.sum(), 1)
    pb = pb / max(pb.sum(), 1)
    return float(0.5 * np.abs(pa - pb).sum())


def shared_bin_edges(pooled: np.ndarray, max_bins: int = 20) -> np.ndarray:
    """Bin edges for TV: one bin per value when values are few, else quantiles.

    TV is bin-dependent, so every comparison for a motif uses one edge set
    derived from the pooled data across all families -- never per-pair edges.
    """
    values = np.unique(pooled)
    if values.size <= max_bins:
        return np.append(values, values[-1] + 1).astype(float)
    levels = np.linspace(0.0, 1.0, max_bins + 1)
    edges = np.unique(np.quantile(pooled.astype(float), levels))
    edges[-1] = edges[-1] + 1.0
    return edges


# ------------------------------------------------------ H1. P3-local
def _h1_family(
    label: str,
    degrees_for_rep,
    reps: int,
    results: list[dict],
) -> None:
    """One family: null floor by degree class, then designed incidence vs it."""
    print(f"\n  {label}")

    top_degree = int(max(int(degrees_for_rep(r).max()) for r in range(reps)))
    bins = degree_bins(degrees_for_rep(0), reps)
    labels = [bin_label(lo, hi, top_degree) for lo, hi in bins]

    # --- null replicates: per-node floor, pooled by degree bin
    floor_obs: dict[int, dict[int, list[int]]] = {
        length: {b: [] for b in range(len(bins))} for _, length in CYCLE_TARGETS
    }
    null_totals: dict[int, list[int]] = {length: [] for _, length in CYCLE_TARGETS}
    for rep in range(reps):
        degrees = degrees_for_rep(rep)
        graph = ca.build(N, degrees, (), seed=H1_NULL_SEED_BASE + rep)
        if graph is None:
            continue
        csr = graph.to_csr()
        bin_of = np.full(N, -1, dtype=np.int_)
        for b, (lo, hi) in enumerate(bins):
            bin_of[(degrees >= lo) & (degrees <= hi)] = b
        for _, length in CYCLE_TARGETS:
            per_node = cycles_per_node(csr, length)
            null_totals[length].append(int(per_node.sum()) // length)
            for b in range(len(bins)):
                floor_obs[length][b].extend(per_node[bin_of == b].tolist())

    floor = {
        length: [
            float(np.mean(floor_obs[length][b])) if floor_obs[length][b] else 0.0
            for b in range(len(bins))
        ]
        for _, length in CYCLE_TARGETS
    }

    print(
        f"    {'target':<9}{'deg bin':>9}{'nodes':>7}{'floor/node':>12}"
        f"{'des/node':>10}{'des|part':>10}{'SNR|part':>10}"
        f"{'real/node':>11}{'insitu fl':>11}{'SNR insitu':>11}"
        f"{'part%':>8}{'%all>=5x':>10}{'%part>=5x':>11}"
    )

    for key, length in CYCLE_TARGETS:
        designed_all: list[np.ndarray] = []
        realized_all: list[np.ndarray] = []
        degrees_all: list[np.ndarray] = []
        realized_totals: list[int] = []
        designed_m: list[int] = []
        for rep in range(reps):
            degrees = degrees_for_rep(rep)
            participation = capped_participation(
                key, degrees, H1_PARTICIPATION_SEED_BASE + rep
            )
            seq = prescribed_cycle_sequence(key, participation)
            graph = ca.build(N, degrees, (seq,), seed=H1_BUILD_SEED_BASE + rep)
            if graph is None:
                continue
            per_node = cycles_per_node(graph.to_csr(), length)
            designed_all.append(participation)
            realized_all.append(per_node)
            degrees_all.append(degrees)
            realized_totals.append(int(per_node.sum()) // length)
            designed_m.append(int(participation.sum()) // length)
        if not designed_all:
            print(f"    {key:<9}  ALL BUILDS FAILED")
            continue

        designed = np.concatenate(designed_all)
        realized = np.concatenate(realized_all)
        degs = np.concatenate(degrees_all)

        # The by-product term ACTUALLY present in the built graph, as opposed
        # to the null family's floor. They are not the same number: imposing
        # structure consumes stubs, which moves the residual random graph, and
        # in a heterogeneous family it moves it upward on the hubs (sec 6.3's
        # additivity ratio drifting from 1.00 to 1.28 is the global shadow of
        # this). SNR insitu is the ratio a message-passing model actually
        # faces at that node.
        insitu = np.maximum(realized - designed, 0)

        clears = np.zeros(designed.size, dtype=bool)
        clears_insitu = np.zeros(designed.size, dtype=bool)
        rows = []
        for b, (lo, hi) in enumerate(bins):
            mask = (degs >= lo) & (degs <= hi)
            f = floor[length][b]
            bar = SNR_BAR * f
            # A node with zero designed incidence never clears, whatever the
            # floor is; a zero floor makes any positive incidence clear.
            in_bin_clear = mask & (designed > 0) & (designed >= bar)
            clears |= in_bin_clear
            f_insitu = float(insitu[mask].mean()) if mask.any() else 0.0
            in_bin_clear_insitu = (
                mask & (designed > 0) & (designed >= SNR_BAR * f_insitu)
            )
            clears_insitu |= in_bin_clear_insitu
            part = mask & (designed > 0)
            n_bin = int(mask.sum())
            n_part = int(part.sum())
            des_part = float(designed[part].mean()) if n_part else 0.0
            snr_part = (des_part / f) if f > 0 else float("inf")
            snr_insitu = (des_part / f_insitu) if f_insitu > 0 else float("inf")
            row = {
                "target": key,
                "degree_bin": labels[b],
                "degree_lo": lo,
                "degree_hi": hi,
                "node_observations": n_bin,
                "floor_per_node": f,
                "designed_per_node": float(designed[mask].mean()) if n_bin else 0.0,
                "designed_per_participating_node": des_part,
                "snr_participating": snr_part,
                "realized_per_node": float(realized[mask].mean()) if n_bin else 0.0,
                "insitu_floor_per_node": f_insitu,
                "snr_participating_insitu": snr_insitu,
                "participating_fraction": (n_part / n_bin) if n_bin else 0.0,
                "frac_all_ge_5x": (int(in_bin_clear.sum()) / n_bin) if n_bin else 0.0,
                "frac_participating_ge_5x": (
                    (int(in_bin_clear.sum()) / n_part) if n_part else 0.0
                ),
                "frac_participating_ge_5x_insitu": (
                    (int(in_bin_clear_insitu.sum()) / n_part) if n_part else 0.0
                ),
            }
            rows.append(row)
            snr_s = "inf" if not np.isfinite(snr_part) else f"{snr_part:.1f}"
            snr_i_s = "inf" if not np.isfinite(snr_insitu) else f"{snr_insitu:.1f}"
            print(
                f"    {key:<9}{labels[b]:>9}{n_bin:>7}{f:>12.3f}"
                f"{row['designed_per_node']:>10.3f}{des_part:>10.3f}{snr_s:>10}"
                f"{row['realized_per_node']:>11.3f}{f_insitu:>11.3f}{snr_i_s:>11}"
                f"{row['participating_fraction']:>7.1%} "
                f"{row['frac_all_ge_5x']:>9.1%}"
                f"{row['frac_participating_ge_5x']:>11.1%}"
            )

        participating = designed > 0
        overall_all = float(clears.mean())
        overall_part = float(clears[participating].mean()) if participating.any() else 0
        overall_part_insitu = (
            float(clears_insitu[participating].mean()) if participating.any() else 0
        )
        global_floor = float(np.mean(null_totals[length])) if null_totals[length] else 0
        global_designed = float(np.mean(designed_m))
        print(
            f"    {key:<9}{'ALL':>9}{designed.size:>7}{'':>12}{'':>10}{'':>10}"
            f"{'':>10}{'':>11}{'':>11}{'':>11}{participating.mean():>7.1%} "
            f"{overall_all:>9.1%}{overall_part:>11.1%}"
        )
        results.append(
            {
                "family": label,
                "target": key,
                "cycle_length": length,
                "bins": rows,
                "overall_frac_all_ge_5x": overall_all,
                "overall_frac_participating_ge_5x": overall_part,
                "overall_frac_participating_ge_5x_insitu": overall_part_insitu,
                "participating_fraction": float(participating.mean()),
                "global_designed_M": global_designed,
                "global_floor": global_floor,
                "global_signal_to_floor": (
                    global_designed / global_floor if global_floor else float("inf")
                ),
                "global_realized": float(np.mean(realized_totals)),
                "replicates": len(designed_all),
            }
        )


def exp_h1_local_snr() -> list[dict]:
    print("\n" + "=" * 78)
    print("H1. P3-LOCAL: per-node incidence vs the DEGREE-CONDITIONAL floor")
    print("=" * 78)
    print(
        f"  n={N}, participation Pois({LAM:.0f}) clipped to the degree cap"
        " (the generator's own default),\n"
        "  then PRESCRIBED via orbit_counts so the designed side is exact."
        f" {REPS_H1} replicates.\n"
        f"  Null floor seeds {H1_NULL_SEED_BASE}+rep, builds"
        f" {H1_BUILD_SEED_BASE}+rep, participation"
        f" {H1_PARTICIPATION_SEED_BASE}+rep,\n"
        f"  degrees {ca.DEGREE_SEED_BASE}+rep. Bar: designed incidence >="
        f" {SNR_BAR:.0f}x the node's own degree-class floor."
    )
    results: list[dict] = []

    regular = np.full(N, REGULAR_DEGREE, dtype=np.int_)
    _h1_family(
        f"H1a. REGULAR degree {REGULAR_DEGREE} (sec 6.2's low-floor family)",
        lambda _rep: regular,
        REPS_H1,
        results,
    )
    print(
        "      NOTE: a degree-regular family has ONE degree class, so"
        " stratification is\n      vacuous here by construction -- and the"
        " per-node floor is then the global\n      floor divided evenly, so"
        " local SNR reproduces global S/F. That is the\n      control: it is"
        " only in the heterogeneous family below that the two part."
    )

    _h1_family(
        f"H1b. HETEROGENEOUS C-model, degrees 2*Pois({LAM:.0f}) (sec 6.3's family)",
        lambda rep: ca.cmodel_degrees(N, LAM, rep),
        REPS_H1,
        results,
    )

    print("\n  Global S/F (context) vs the local picture:")
    print(
        f"    {'family':<34}{'target':<10}{'global S/F':>12}"
        f"{'%all>=5x':>10}{'%part>=5x':>11}{'%part insitu':>14}"
    )
    for r in results:
        print(
            f"    {r['family'][:33]:<34}{r['target']:<10}"
            f"{r['global_signal_to_floor']:>12.2f}"
            f"{r['overall_frac_all_ge_5x']:>10.1%}"
            f"{r['overall_frac_participating_ge_5x']:>11.1%}"
            f"{r['overall_frac_participating_ge_5x_insitu']:>14.1%}"
        )
    print(
        "\n  'floor/node' is the NULL family's per-node count in that degree"
        " class; 'insitu fl'\n  is the by-product term left in the BUILT"
        " graph (realized - designed). The two\n  part in the heterogeneous"
        " family because imposing structure consumes stubs and\n  moves the"
        " residual random graph -- upward on the hubs. 'SNR insitu' and"
        "\n  '%part insitu' are the stricter, more honest reading of"
        " P3-local."
    )
    return results


# ------------------------------------------------------ H2. P8, the leak check
H2_FAMILIES = (
    ("Null", None),
    ("G_square", "square"),
    ("G_C5", "pentagon"),
    ("G_C6", "hexagon"),
)

# Which motif each family DESIGNS. A designed motif is not "held fixed" across
# a pair containing that family -- it is the contrast, and a large distance
# there is the point rather than a leak.
DESIGNED_MOTIF = {
    "Null": None,
    "G_square": "cycle4",
    "G_C5": "cycle5",
    "G_C6": "cycle6",
}

H2_MOTIFS = (
    "cycle3",
    "cycle4",
    "cycle5",
    "cycle6",
    "path",
    "star",
    "paw",
    "diamond",
    "complete",
)


def _h2_profiles(reps: int) -> dict[str, dict[str, list[np.ndarray]]]:
    """Per-node incidence profiles, per family, per motif, per replicate."""
    profiles: dict[str, dict[str, list[np.ndarray]]] = {
        label: {motif: [] for motif in H2_MOTIFS} for label, _ in H2_FAMILIES
    }
    for label, key in H2_FAMILIES:
        for rep in range(reps):
            degrees = ca.cmodel_degrees(N, LAM, rep)
            seqs: tuple[SubgraphSequence, ...] = (
                ()
                if key is None
                else (SubgraphSequence(subgraph=ca.SG[key], distribution=poisson(LAM)),)
            )
            graph = ca.build(N, degrees, seqs, seed=H2_SEED_BASE + rep)
            if graph is None:
                continue
            csr = graph.to_csr()
            for length in (3, 4, 5, 6):
                profiles[label][f"cycle{length}"].append(cycles_per_node(csr, length))
            order_four = order_four_per_node(csr)
            for motif in ("path", "star", "paw", "diamond", "complete"):
                profiles[label][motif].append(order_four[motif])
    return profiles


def exp_h2_leak() -> dict:
    print("\n" + "=" * 78)
    print("H2. P8: do the per-node incidence profiles of a matched pair agree?")
    print("=" * 78)
    print(
        f"  C-model family {{Null, G_square, G_C5, G_C6}}, n={N},"
        f" degrees 2*Pois({LAM:.0f}),\n"
        f"  {REPS_H2} replicates, build seeds {H2_SEED_BASE}+rep --"
        " the same builds as sec 6.3/6.4.\n"
        "  TV on shared bins (per-value where values are few, else 20"
        " quantile bins),\n"
        "  KS as a bin-free cross-check. Baseline: TV between two draws of"
        " the SAME family."
    )
    t0 = time.time()
    profiles = _h2_profiles(REPS_H2)
    print(f"  ({time.time() - t0:.1f}s to build and profile)")

    labels = [label for label, _ in H2_FAMILIES]
    edges: dict[str, np.ndarray] = {}
    for motif in H2_MOTIFS:
        pooled = np.concatenate(
            [np.concatenate(profiles[label][motif]) for label in labels]
        )
        edges[motif] = shared_bin_edges(pooled)

    # within-family scatter, per family per motif
    within: dict[str, dict[str, dict[str, float]]] = {}
    for label in labels:
        within[label] = {}
        for motif in H2_MOTIFS:
            reps_list = profiles[label][motif]
            vals = [
                total_variation(reps_list[i], reps_list[j], edges[motif])
                for i in range(len(reps_list))
                for j in range(i + 1, len(reps_list))
            ]
            ks_vals = [
                float(ks_2samp(reps_list[i], reps_list[j]).statistic)
                for i in range(len(reps_list))
                for j in range(i + 1, len(reps_list))
            ]
            within[label][motif] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals)) if vals else 0.0,
                "max": float(np.max(vals)) if vals else 0.0,
                "ks_mean": float(np.mean(ks_vals)) if ks_vals else 0.0,
                "ks_max": float(np.max(ks_vals)) if ks_vals else 0.0,
                "pairs": len(vals),
            }

    print("\n  Within-family replicate scatter (mean TV over replicate pairs):")
    header = "".join(f"{m:>10}" for m in H2_MOTIFS)
    print(f"    {'family':<11}{header}")
    for label in labels:
        print(
            f"    {label:<11}"
            + "".join(f"{within[label][m]['mean']:>10.4f}" for m in H2_MOTIFS)
        )

    print(
        "\n  Between-family distance. 'held' = motif neither family designs."
        "\n  z = (TV_between - mean TV_within) / std TV_within, pooling both"
        " families' scatter."
    )
    print(
        f"    {'pair':<22}{'motif':<9}{'held':>6}{'TV':>9}{'TV_null':>9}"
        f"{'z':>8}{'KS':>8}{'KS_null':>9}{'KS p':>10}{'verdict':>12}"
    )
    rows = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            pair = f"{a} vs {b}"
            for motif in H2_MOTIFS:
                pa, pb = profiles[a][motif], profiles[b][motif]
                if not pa or not pb:
                    continue
                cross = [total_variation(x, y, edges[motif]) for x in pa for y in pb]
                tv = float(np.mean(cross))
                w_mean = 0.5 * (within[a][motif]["mean"] + within[b][motif]["mean"])
                w_std = 0.5 * (within[a][motif]["std"] + within[b][motif]["std"])
                if w_std > 0:
                    z = (tv - w_mean) / w_std
                else:
                    # Zero within-family scatter is degenerate, not infinitely
                    # sensitive: it means the motif is absent (or constant) in
                    # every replicate of both families. Only a between-family
                    # distance that is itself non-zero can be a leak there.
                    z = float("inf") if tv > w_mean + 1e-12 else 0.0
                ks = ks_2samp(np.concatenate(pa), np.concatenate(pb))
                ks_within = 0.5 * (
                    within[a][motif]["ks_mean"] + within[b][motif]["ks_mean"]
                )
                held = motif not in (DESIGNED_MOTIF[a], DESIGNED_MOTIF[b])
                if not held:
                    verdict = "TARGET"
                elif z <= 3:
                    verdict = "no leak"
                else:
                    verdict = "LEAK"
                rows.append(
                    {
                        "pair": pair,
                        "family_a": a,
                        "family_b": b,
                        "motif": motif,
                        "held_fixed": held,
                        "tv_between": tv,
                        "tv_within_mean": w_mean,
                        "tv_within_std": w_std,
                        "z": float(z) if np.isfinite(z) else None,
                        "z_infinite": not np.isfinite(z),
                        "ks_statistic": float(ks.statistic),
                        "ks_within_mean": float(ks_within),
                        "ks_pvalue": float(ks.pvalue),
                        "verdict": verdict,
                    }
                )
                z_s = "inf" if not np.isfinite(z) else f"{z:.1f}"
                print(
                    f"    {pair:<22}{motif:<9}{'y' if held else 'n':>6}{tv:>9.4f}"
                    f"{w_mean:>9.4f}{z_s:>8}{ks.statistic:>8.4f}"
                    f"{ks_within:>9.4f}{ks.pvalue:>10.2e}{verdict:>12}"
                )

    print(
        "\n  KS is pooled over all replicates (12 x 1000 nodes), so its"
        " p-value is\n  over-powered here -- it rejects at differences well"
        " inside replicate scatter.\n  Read KS against the KS_null column"
        " (same-family replicate pairs), not against p."
    )

    leaks = [r for r in rows if r["held_fixed"] and r["verdict"] == "LEAK"]
    clean = [r for r in rows if r["held_fixed"] and r["verdict"] == "no leak"]
    print(
        f"\n  {len(clean)} held-fixed (pair, motif) comparisons within scatter,"
        f" {len(leaks)} outside it."
    )
    if leaks:
        print("  Leaking comparisons, worst first:")
        for r in sorted(leaks, key=lambda r: -(r["z"] if r["z"] is not None else 1e9))[
            :20
        ]:
            z_s = "inf" if r["z"] is None else f"{r['z']:.1f}"
            print(
                f"    {r['pair']:<22}{r['motif']:<9}TV={r['tv_between']:.4f}"
                f" (null {r['tv_within_mean']:.4f}), z={z_s},"
                f" KS={r['ks_statistic']:.4f}"
            )
    return {"within_family_scatter": within, "pairs": rows}


# ------------------------------------------------- H3. the placement experiment
def exp_h3_placement() -> dict:
    print("\n" + "=" * 78)
    print("H3. PLACEMENT: can hub-avoidance rescue local SNR where global fails?")
    print("=" * 78)
    print(
        f"  Heterogeneous pentagon family (sec 6.3): n={N},"
        f" degrees 2*Pois({LAM:.0f}), rate Pois({LAM:.0f}).\n"
        "  Globally the pentagon sits BELOW its own floor (0.8x, sec 6.3)."
        " Hypothesis: local SNR\n  at participating nodes still clears 5x if"
        " placement avoids the hubs.\n"
        f"  {REPS_H3} replicates. Null seeds {H3_NULL_SEED_BASE}+rep, builds"
        f" {H3_BUILD_SEED_BASE}+rep,\n  participation"
        f" {H3_PARTICIPATION_SEED_BASE}+rep, placement"
        f" {H3_PLACEMENT_SEED_BASE}+rep, degrees {ca.DEGREE_SEED_BASE}+rep.\n"
        "  Three arms:\n"
        "    capped      -- the generator's default draw (ticket 003's cap,"
        " hub-correlated)\n"
        "    neutral     -- degree-neutral water-fill, prescribed via"
        " split_deterministic\n"
        "    low-degree  -- maximally hub-AVOIDING; not neutral, but it"
        " bounds what any\n                   placement rule could buy"
    )
    length = 5
    cardinality = 2

    degrees_by_rep = {rep: ca.cmodel_degrees(N, LAM, rep) for rep in range(REPS_H3)}
    bins = degree_bins(degrees_by_rep[0], REPS_H3)
    top_degree = max(int(d.max()) for d in degrees_by_rep.values())
    labels = [bin_label(lo, hi, top_degree) for lo, hi in bins]

    # --- degree-conditional pentagon floor
    floor_obs: dict[int, list[int]] = {b: [] for b in range(len(bins))}
    null_totals: list[int] = []
    for rep in range(REPS_H3):
        degrees = degrees_by_rep[rep]
        graph = ca.build(N, degrees, (), seed=H3_NULL_SEED_BASE + rep)
        if graph is None:
            continue
        per_node = cycles_per_node(graph.to_csr(), length)
        null_totals.append(int(per_node.sum()) // length)
        for b, (lo, hi) in enumerate(bins):
            floor_obs[b].extend(per_node[(degrees >= lo) & (degrees <= hi)].tolist())
    floor = [
        float(np.mean(floor_obs[b])) if floor_obs[b] else 0.0 for b in range(len(bins))
    ]
    global_floor = float(np.mean(null_totals)) if null_totals else 0.0

    print(f"\n  Degree-conditional pentagon floor (null, {len(null_totals)} reps):")
    print(f"    {'deg bin':>9}{'floor/node':>12}{'cap':>6}{'needed for 5x':>15}")
    ceiling_rows = []
    for b, (lo, hi) in enumerate(bins):
        cap_hi = hi // cardinality
        need = SNR_BAR * floor[b]
        feasible = need <= cap_hi
        ceiling_rows.append(
            {
                "degree_bin": labels[b],
                "floor_per_node": floor[b],
                "max_cap_in_bin": cap_hi,
                "participations_needed_for_5x": need,
                "feasible_at_any_placement": bool(feasible),
            }
        )
        print(
            f"    {labels[b]:>9}{floor[b]:>12.3f}{cap_hi:>6}{need:>13.2f}"
            f"{'  ok' if feasible else '  IMPOSSIBLE'}"
        )
    print(
        "    ('cap' is floor(k/2), the most participations the degree budget"
        " allows in\n     the top class of the bin; 'needed for 5x' is 5x the"
        " bin's floor. Where the\n     second exceeds the first, NO placement"
        " can make those nodes clear the bar.)"
    )

    arms = ("capped", "neutral", "low-degree")
    per_arm: dict[str, dict] = {}
    print(
        f"\n  {'arm':<11}{'M':>7}{'part nodes':>12}{'corr(v,k)':>11}"
        f"{'mean k|part':>13}{'%all>=5x':>10}{'%part>=5x':>11}"
        f"{'%part insitu':>14}{'global S/F':>12}{'realized':>10}{'insitu S/F':>12}"
    )
    for arm in arms:
        designed_all, degrees_all, realized_all = [], [], []
        corrs, m_vals, realized_totals = [], [], []
        for rep in range(REPS_H3):
            degrees = degrees_by_rep[rep]
            capped = capped_participation(
                "pentagon", degrees, H3_PARTICIPATION_SEED_BASE + rep
            )
            total = int(capped.sum())  # every arm designs the SAME M
            if arm == "capped":
                placement = capped
            elif arm == "neutral":
                placement = water_fill_placement(
                    total, degrees, cardinality, H3_PLACEMENT_SEED_BASE + rep
                )
            else:
                placement = low_degree_first_placement(
                    total, degrees, cardinality, H3_PLACEMENT_SEED_BASE + rep
                )
            if int(placement.sum()) != total:
                print(
                    f"    NOTE: {arm} rep {rep} could only place"
                    f" {int(placement.sum())} of {total} participations"
                    " (degree budget exhausted); arm is not M-matched here."
                )
            seq = prescribed_cycle_sequence("pentagon", placement)
            graph = ca.build(N, degrees, (seq,), seed=H3_BUILD_SEED_BASE + rep)
            if graph is None:
                print(f"    {arm} rep {rep}: BUILD FAILED")
                continue
            per_node = cycles_per_node(graph.to_csr(), length)
            designed_all.append(placement)
            degrees_all.append(degrees)
            realized_all.append(per_node)
            corrs.append(float(np.corrcoef(placement, degrees)[0, 1]))
            m_vals.append(int(placement.sum()) // length)
            realized_totals.append(int(per_node.sum()) // length)
        if not designed_all:
            print(f"  {arm:<11} ALL BUILDS FAILED")
            continue

        designed = np.concatenate(designed_all)
        degs = np.concatenate(degrees_all)
        realized = np.concatenate(realized_all)
        insitu = np.maximum(realized - designed, 0)
        clears = np.zeros(designed.size, dtype=bool)
        clears_insitu = np.zeros(designed.size, dtype=bool)
        snr = np.zeros(designed.size, dtype=float)
        bin_rows = []
        for b, (lo, hi) in enumerate(bins):
            mask = (degs >= lo) & (degs <= hi)
            f = floor[b]
            if f > 0:
                snr[mask] = designed[mask] / f
            else:
                snr[mask] = np.where(designed[mask] > 0, np.inf, 0.0)
            clears |= mask & (designed > 0) & (designed >= SNR_BAR * f)
            f_insitu = float(insitu[mask].mean()) if mask.any() else 0.0
            clears_insitu |= mask & (designed > 0) & (designed >= SNR_BAR * f_insitu)
            part = mask & (designed > 0)
            n_bin, n_part = int(mask.sum()), int(part.sum())
            bin_rows.append(
                {
                    "degree_bin": labels[b],
                    "node_observations": n_bin,
                    "floor_per_node": f,
                    "insitu_floor_per_node": f_insitu,
                    "designed_per_participating_node": (
                        float(designed[part].mean()) if n_part else 0.0
                    ),
                    "participating_fraction": (n_part / n_bin) if n_bin else 0.0,
                    "frac_participating_ge_5x": (
                        float((mask & (designed > 0) & (designed >= SNR_BAR * f)).sum())
                        / n_part
                        if n_part
                        else 0.0
                    ),
                    "frac_participating_ge_5x_insitu": (
                        float(
                            (
                                mask & (designed > 0) & (designed >= SNR_BAR * f_insitu)
                            ).sum()
                        )
                        / n_part
                        if n_part
                        else 0.0
                    ),
                    "realized_per_node": float(realized[mask].mean()) if n_bin else 0.0,
                }
            )
        part_mask = designed > 0
        snr_part = snr[part_mask]
        snr_part = snr_part[np.isfinite(snr_part)]
        global_sf = float(np.mean(m_vals)) / global_floor if global_floor else np.inf
        per_arm[arm] = {
            "designed_M": float(np.mean(m_vals)),
            "participating_nodes": float(part_mask.mean() * N),
            "participating_fraction": float(part_mask.mean()),
            "corr_participation_degree": float(np.mean(corrs)),
            "mean_degree_of_participants": float(degs[part_mask].mean()),
            "frac_all_ge_5x": float(clears.mean()),
            "frac_participating_ge_5x": float(clears[part_mask].mean()),
            "frac_participating_ge_5x_insitu": float(clears_insitu[part_mask].mean()),
            "insitu_floor_total": (
                float(np.mean(realized_totals)) - float(np.mean(m_vals))
            ),
            "realized_over_null_floor": (
                float(np.mean(realized_totals)) / global_floor if global_floor else None
            ),
            "global_signal_to_floor": global_sf,
            "global_floor": global_floor,
            "global_realized": float(np.mean(realized_totals)),
            "snr_participating_quantiles": {
                q: float(np.quantile(snr_part, q / 100))
                for q in (10, 25, 50, 75, 90, 99)
            },
            "snr_participating_mean": float(snr_part.mean()),
            "snr_participating_max": float(snr_part.max()),
            "bins": bin_rows,
            "replicates": len(designed_all),
        }
        print(
            f"  {arm:<11}{np.mean(m_vals):>7.0f}{part_mask.mean() * N:>12.0f}"
            f"{np.mean(corrs):>11.3f}{degs[part_mask].mean():>13.2f}"
            f"{clears.mean():>10.1%}{clears[part_mask].mean():>11.1%}"
            f"{clears_insitu[part_mask].mean():>14.1%}"
            f"{global_sf:>12.2f}{np.mean(realized_totals):>10.1f}"
            f"{np.mean(m_vals) / max(per_arm[arm]['insitu_floor_total'], 1e-9):>12.2f}"
        )

    print("\n  Per-node SNR at PARTICIPATING nodes (designed / degree-class floor):")
    print(
        f"    {'arm':<11}{'mean':>8}{'p10':>8}{'p25':>8}{'p50':>8}"
        f"{'p75':>8}{'p90':>8}{'p99':>8}{'max':>8}"
    )
    for arm in arms:
        if arm not in per_arm:
            continue
        q = per_arm[arm]["snr_participating_quantiles"]
        print(
            f"    {arm:<11}{per_arm[arm]['snr_participating_mean']:>8.2f}"
            + "".join(f"{q[p]:>8.2f}" for p in (10, 25, 50, 75, 90, 99))
            + f"{per_arm[arm]['snr_participating_max']:>8.2f}"
        )

    print(
        "\n  corr(v,k) is a poor read on 'degree-neutral': the degree BUDGET"
        " forces\n  v_i <= k_i/2, so no feasible placement can have r = 0."
        " The water-fill arm's\n  higher r is an artefact of that floor"
        " (every eligible node gets 1, only\n  k>=4 nodes can take a 2nd) --"
        " read 'mean k|part' instead, which is flat\n  across capped and"
        " neutral and drops only for the hub-avoiding arm."
    )
    print(
        "\n  'insitu S/F' is designed M over the by-product term left in the"
        " BUILT graph\n  (realized - designed), not over the null family's"
        " floor. Saturating the\n  periphery pushes every free stub onto the"
        " hubs, so the hub-avoiding arm\n  builds a much denser core and"
        " inflates its OWN floor -- a cost the null-\n  referenced columns"
        " do not show."
    )

    print("\n  Per-degree-bin, fraction of PARTICIPATING nodes clearing 5x:")
    print(f"    {'arm':<11}" + "".join(f"{label:>11}" for label in labels))
    for arm in arms:
        if arm not in per_arm:
            continue
        print(
            f"    {arm:<11}"
            + "".join(
                f"{row['frac_participating_ge_5x']:>11.1%}"
                for row in per_arm[arm]["bins"]
            )
        )

    print("\n  Per-degree-bin, IN-SITU floor per node (realized - designed):")
    print(f"    {'arm':<11}" + "".join(f"{label:>11}" for label in labels))
    for arm in arms:
        if arm not in per_arm:
            continue
        print(
            f"    {arm:<11}"
            + "".join(
                f"{row['insitu_floor_per_node']:>11.3f}" for row in per_arm[arm]["bins"]
            )
        )

    print(
        "\n  Per-degree-bin, fraction of PARTICIPATING nodes clearing 5x the"
        " IN-SITU floor:"
    )
    print(f"    {'arm':<11}" + "".join(f"{label:>11}" for label in labels))
    for arm in arms:
        if arm not in per_arm:
            continue
        print(
            f"    {arm:<11}"
            + "".join(
                f"{row['frac_participating_ge_5x_insitu']:>11.1%}"
                for row in per_arm[arm]["bins"]
            )
        )

    best = max(per_arm, key=lambda a: per_arm[a]["frac_participating_ge_5x"])
    rescued = per_arm[best]["frac_participating_ge_5x"] >= 0.5
    print(
        f"\n  VERDICT: hypothesis {'CONFIRMED' if rescued else 'REFUTED'}."
        f" Best arm '{best}' clears 5x at"
        f" {per_arm[best]['frac_participating_ge_5x']:.1%} of participating"
        f" nodes\n  (capped default:"
        f" {per_arm['capped']['frac_participating_ge_5x']:.1%}), against a"
        f" global S/F of"
        f" {per_arm['capped']['global_signal_to_floor']:.2f}x."
    )
    print(
        "\n  Why it is refuted, and why it is not a tuning failure: the"
        " 'needed for 5x'\n  column above is a CEILING argument. At k>=4 the"
        " pentagon floor already exceeds\n  one fifth of the degree budget"
        f" ({SNR_BAR:.0f}x floor > floor(k/2)), so no placement rule"
        " -- and no\n  participation sequence that fits the degree budget --"
        " can make those nodes clear\n  the bar. Only k=2 nodes can, and"
        " every arm already saturates them. Placement\n  moves WHICH nodes"
        " participate; it cannot move the floor, which is set by degree."
    )
    low = per_arm.get("low-degree")
    if low is not None:
        print(
            "\n  One trap in the hub-avoiding arm. Against the IN-SITU floor"
            f" it looks rescued\n  ({low['frac_participating_ge_5x_insitu']:.1%}"
            " of participants clear 5x, vs"
            f" {per_arm['capped']['frac_participating_ge_5x_insitu']:.1%}"
            " capped), because\n  saturating the periphery locks those nodes"
            " inside their pentagons with no free\n  stubs left. But the"
            " stubs have to go somewhere: they pile onto the hubs, and the"
            f"\n  graph realizes {low['global_realized']:.0f} pentagons"
            f" against a null floor of {global_floor:.0f}"
            f" ({low['realized_over_null_floor']:.1f}x).\n  That enrichment"
            " is by-product, not design -- it would separate the family from"
            " the\n  null on a global count while carrying no designed"
            " signal, which is a P4 failure\n  bought with a P3-local win."
            " Hub-avoidance is not free."
        )
    return {
        "degree_conditional_floor": ceiling_rows,
        "global_floor": global_floor,
        "arms": per_arm,
        "best_arm": best,
        "hypothesis_confirmed": bool(rescued),
    }


def main() -> None:
    t0 = time.time()
    out = {
        "parameters": {
            "n": N,
            "lam": LAM,
            "regular_degree": REGULAR_DEGREE,
            "snr_bar": SNR_BAR,
            "min_bin_observations": MIN_BIN_OBS,
            "reps_h1": REPS_H1,
            "reps_h2": REPS_H2,
            "reps_h3": REPS_H3,
            "seeds": {
                "degrees": f"{ca.DEGREE_SEED_BASE}+rep (control_audit)",
                "h1_null_builds": f"{H1_NULL_SEED_BASE}+rep",
                "h1_signal_builds": f"{H1_BUILD_SEED_BASE}+rep",
                "h1_participation": f"{H1_PARTICIPATION_SEED_BASE}+rep",
                "h2_builds": f"{H2_SEED_BASE}+rep (reuses section C/F)",
                "h3_null_builds": f"{H3_NULL_SEED_BASE}+rep",
                "h3_signal_builds": f"{H3_BUILD_SEED_BASE}+rep",
                "h3_participation": f"{H3_PARTICIPATION_SEED_BASE}+rep",
                "h3_placement": f"{H3_PLACEMENT_SEED_BASE}+rep",
            },
            "caps_applied": (
                "none -- no n, replicate count or cycle length was reduced for runtime"
            ),
        },
        "h1_local_snr": exp_h1_local_snr(),
        "h2_leak": exp_h2_leak(),
        "h3_placement": exp_h3_placement(),
    }
    out["parameters"]["runtime_seconds"] = round(time.time() - t0, 1)
    dest = Path(__file__).parent / "audit_section_h_results.json"
    dest.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nruntime {out['parameters']['runtime_seconds']}s")
    print(f"wrote {dest}")


if __name__ == "__main__":
    main()
