# Claim 4 — Higher-order structure: what is and is not controlled

**With degrees and clustering pinned, real freedom remains: which four-node
shapes appear, how shapes are placed across nodes, how local clustering varies
with degree. That residual freedom is the experimental variable. It is what a
benchmark varies.**

But "free" cuts both ways. What you do not control still moves. Residual
freedom is an asset only once it is measured; until then it is a confound.
Every number below comes from the heterogeneous cycle families of the audit
($n = 1000$, degrees $2 \times \mathrm{Pois}(2)$, 12 builds per family unless
stated) and the four-node census (4 builds per family).

## The assortativity confound

The degree cap that makes sampled builds reliable ties participation to
degree. Shapes land preferentially on hubs, and every subgraph family comes
out degree-assortative without anyone asking: mean assortativity $+0.087$,
$+0.101$, $+0.090$ and $+0.096$ for the triangle, square, pentagon and
hexagon families (standard deviation across builds 0.02 to 0.03) against
$-0.012$ (0.024) for the null. The cycle families are matched *to each
other*, but not to the null. Comparing any of them against the null
confounds cycle structure with assortativity.

Placement direction is not the intuitive one. Giving hubs the heavy *roles*
inside an asymmetric shape **lowers** assortativity relative to an even
split: the shape's edges mostly run from heavy roles to light ones, so ranking
roles by degree sends edges across the degree gap. The regression test for
`split_by_degree_rank` (diamond, $n = 300$) asserts a drop of at least 0.05.
What raises assortativity is concentrating *whole shapes* on hubs, which is
what the degree cap quietly does.

## The per-node leak

A pair matched on every global count can still differ per node, and per-node
incidence is precisely what a message-passing network reads. The test: for
each pair of families and each cycle length not designed into either, the
total-variation distance between their per-node incidence distributions is
compared with the spread between replicate builds of one family (66 replicate
pairs). A pair leaks when it sits more than three of those spreads apart.

Measured: the null and the square family agree on every global count yet
differ per node in pentagon and hexagon incidence ($z = 4.4$ on both). Every
cycle-versus-cycle pair stays within two spreads. The
Kolmogorov–Smirnov test is reported alongside; at 12 builds it rejects
almost every comparison and is not used for the verdict. Same lesson as the
assortativity confound, reached independently: leave the null out of matched
pairs.

## Global and local signal decouple

The global 5×-the-floor test and its per-node counterpart disagree in both
directions:

- The square family fails globally at 4.1× (designed 306 against a floor of
  74), yet 71% of its participating nodes clear 5× against their own local
  floor, because the floor lives on hubs and most participants are not hubs.
  It is usable for node-level tasks.
- The triangle family passes globally at 21× (designed 408 against 19),
  while its hub participants (degree 10 and above) do not all clear locally:
  68% do against the null's per-degree floor, and 36% against the in-situ
  floor measured on the family's own builds.

Placement cannot rescue what degree forbids. From degree 4 upward the
heterogeneous pentagon floor already exceeds anything a node's whole degree
budget could buy: a degree-4 node would need 4.6 participations to clear 5×
and can afford 2. And aggressive hub-avoidance backfires. Pushing pentagon
participation onto low-degree nodes leaves the displaced stubs to pile onto
the hubs, which then generate 9.1× the null floor in *accidental* pentagons
(2646 realized against a floor of 289 and a design of 248): global signal
with nothing designed behind it.

## The four-node census

The direct check on what appears unasked-for is to count all six connected
four-node classes. Against the null (paths 47 464, stars 17 590, squares 74,
paws 385, diamonds 1.5, $K_4$ 0):

- In the cycle families the only class that moves is the designed one; the
  paw's share of connected quadruples stays at 0.6% to 0.7%.
- In the diamond and $K_4$ families the paw rises to 7.4% and 15.3% of
  quadruples, by construction: a paw is a triangle plus a pendant edge, so it
  tracks the designed triangles.
- The $K_4$ family also carries 14 induced diamonds against the null's 1.5,
  and the square family 4.3. Those are chords: a plain edge that lands
  across a designed square makes a diamond, and a designed $K_4$ that loses
  nothing is still surrounded by near-misses. Paths and stars fall in the
  dense families (to 29 876 and 9 989 in the $K_4$ family) because the same
  edges are now spent inside shapes.

So nothing appears without a structural reason, but "only the paw moves" is
true of the cycle families alone.

## What this means for a benchmark

Measure clustering *and* assortativity for every family, every time. Judge
node-level datasets locally, not globally. Prescribe placement when it
matters: placement is a choice, not bookkeeping.

## Evidence

Assortativity and the census:
[`control_audit.py`](evidence/control_audit.py) →
[`control_audit_results.json`](evidence/control_audit_results.json),
sections `assortativity` (12 builds) and `order_four` (4 builds). Local
signal-to-floor, the leak test and the placement arms:
[`audit_section_h.py`](evidence/audit_section_h.py) →
[`audit_section_h_results.json`](evidence/audit_section_h_results.json)
(12 builds per arm). The placement-direction result is a regression test,
`test_degree_rank_split_lowers_realized_assortativity` in
`tests/graphs/configuration_model/test_subgraph_sequence.py`, not an audit
script.

!!! info "Reference"
    Degree assortativity: Newman, M. E. J., *Assortative mixing in
    networks*, Phys. Rev. Lett. 89 (2002) 208701. The subgraph roles that
    make placement a design variable: Karrer, B. and Newman, M. E. J., Phys.
    Rev. E 82 (2010) 066118.

## Where in the code

- `split_deterministic`, `split_by_degree_rank` — explicit placement control.
- `cycles_per_node`, `order_four_per_node`, `degree_assortativity` — the
  per-node and census metrics.
- [Metrics API](../api/metrics.md)
