# Claim 4 — Higher-order structures: what is and is not controlled

**With degrees and clustering pinned, real freedom remains: which four-node shapes
appear, how shapes are placed across nodes, how local clustering varies with degree.
That residual freedom is the experimental variable — it is what a benchmark varies.**

But "free" cuts both ways. What you do not control still moves. Residual freedom is an
asset only once it is measured; until then it is a confound. The measured confounds:

## The assortativity confound

The degree cap that makes sampled builds reliable ties participation to degree. Shapes
land preferentially on hubs, and every subgraph family comes out at degree
assortativity ≈ +0.09 against the null's ≈ 0 — without anyone asking. The cycle
families are matched *to each other*, but not to the null. Comparing any of them
against the null confounds cycle structure with assortativity.

Placement direction is not the intuitive one. Giving hubs the heavy *roles* inside an
asymmetric shape **lowers** assortativity (−0.12 against +0.03): the shape's edges
mostly run from heavy roles to light ones, so ranking roles by degree sends edges
across the degree gap. What raises assortativity is concentrating *whole shapes* on
hubs — which is what the degree cap quietly does.

## The per-node leak

A pair matched on every global count can still differ per node — and per-node
incidence is precisely what a message-passing network reads. Measured: the null and
the square family agree on every global count yet differ, per node, in pentagon and
hexagon incidence. The cycle families compared among themselves stay clean. Same
lesson as the assortativity confound, reached independently: leave the null out of
matched pairs.

## Global and local signal decouple

The global 5×-the-floor test and its per-node counterpart disagree in both directions:

- The square family fails globally at 4.1× — yet 70% of its participating nodes clear
  5× against their own local floor, because the floor lives on hubs and most
  participants are not hubs. Usable for node-level tasks.
- The triangle family passes globally at 21× — while a third of its hub participants
  drown locally.

Placement cannot rescue what degree forbids: from degree 4 upward the heterogeneous
pentagon floor already exceeds anything a node's whole degree budget could buy, so no
placement rule helps. Aggressive hub-avoidance backfires — displaced stubs pile onto
the hubs and generate nine times the floor in *accidental* pentagons: global signal
with nothing designed behind it.

## The four-node census

The direct check that nothing unasked-for appears: counting all four-node classes
shows the only class that moves above its floor (the paw) moves by construction — it
is a triangle plus a pendant edge, so it tracks the designed triangles.

## What this means for a benchmark

Measure clustering *and* assortativity for every family, every time. Judge node-level
datasets locally, not globally. Prescribe placement when it matters — placement is a
choice, not bookkeeping.

## Evidence

Assortativity, the census, orbit splits:
[`control_audit.py`](evidence/control_audit.py) →
[`control_audit_results.json`](evidence/control_audit_results.json). Local
signal-to-floor, matched-pair leakage, placement experiments:
[`audit_section_h.py`](evidence/audit_section_h.py) →
[`audit_section_h_results.json`](evidence/audit_section_h_results.json).

## Where in the code

- `split_deterministic`, `split_by_degree_rank` — explicit placement control.
- `cycles_per_node`, `order_four_per_node`, `degree_assortativity` — the per-node and
  census metrics.
- [Metrics API](../api/metrics.md)
