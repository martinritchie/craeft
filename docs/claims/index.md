# The Claims

*What the generator guarantees, what it assumes, and what it leaves uncontrolled.
This page gives the whole picture. Each claim has its own page with the evidence
behind it, and every number here was produced by the scripts in
[`evidence/`](#9-the-evidence) — the raw results ship alongside them.*

---

## Contents

1. [The idea in one page](#1-the-idea-in-one-page)
2. [Six terms to know](#2-six-terms-to-know)
3. [The four claims](#3-the-four-claims)
4. [The assumptions](#4-the-assumptions)
5. [How the generator works](#5-how-the-generator-works)
6. [What measurement shows](#6-what-measurement-shows)
7. [Rules of thumb for building a dataset](#7-rules-of-thumb-for-building-a-dataset)
8. [Open questions](#8-open-questions)
9. [The evidence](#9-the-evidence)

---

## 1. The idea in one page

The generator builds graphs to order. Hand it a degree sequence, a set of small
subgraph shapes and a count of how many of each shape each node should join, and it
wires a graph that honours all three: every degree exact, the subgraph counts set by
design, the clustering coefficient known before the build begins.

That combination makes controlled experiments on graph structure possible. Two
graphs can share a degree distribution and a clustering coefficient and still differ
in the shapes they are woven from — and the difference matters. It changes how an
epidemic spreads. It changes what a learning algorithm can read from the graph. To
study any effect of this kind you need graph families identical in everything except
the structure under test, and building those families is what the generator is for.

The underlying theory comes from three papers:

- **Ritchie, Berthouze, House & Kiss (2014, JTB)** showed the effect is real:
  four-node structure changes epidemic outcomes even with degrees and clustering held
  fixed. The paper also built the metrics that make the difference measurable.
- **Ritchie, Berthouze & Kiss (2016, JMB)** built the theory: how to describe graphs
  assembled from arbitrary subgraphs — including ones that are not fully connected —
  and how to derive epidemic equations for them automatically.
- **Ritchie, Berthouze & Kiss (2017, JCN)** built the generators: two algorithms that
  construct graphs from subgraph building blocks while preserving an exact degree
  sequence.

One of those generators is the tool this project develops: **orbit-resolved CMA**
(cardinality matching algorithm). Everything below is about what that tool guarantees,
what it assumes, and what it quietly does not control.

One thing to hold onto: the burden of proof is deliberately high. A controlled
comparison is only informative if the contrast is real — if the families differ where
the design says they differ, and nowhere else. So every claim comes in three parts:
what is held exactly, what is held approximately, and what is not controlled at all.

---

## 2. Six terms to know

**Subgraph.** A small shape used as a building block: an edge, a triangle, a square, a
pentagon, a diamond (a square with one diagonal), a complete square $K_4$. The papers
call these motifs.

**Configuration model.** The classic way to build a random graph with a chosen degree
sequence. Give every node a number of half-edges ("stubs") and pair the stubs at random.

**Hyperstub.** The same idea, upgraded. Instead of a half-edge, a node holds a
half-*subgraph* — "one corner of a triangle", say. Pairing hyperstubs assembles whole
shapes rather than single edges.

**Orbit.** The distinct *positions* within a shape. A diamond has two: the tip corners
and the diagonal corners. They cost different numbers of edges (2 vs 3) and sit in
different numbers of triangles (1 vs 2). Treating positions separately — "orbit
resolution" — is what makes the bookkeeping exact rather than exact-on-average.

**Participation.** How many copies of each shape a given node belongs to. This is the
generator's main input. It can be *prescribed* (you say exactly) or *sampled* (drawn
from a distribution, e.g. Poisson).

**The by-product floor.** Shapes also appear by accident. Wire any random graph and
some triangles, squares and hexagons form on their own. The floor is how many you get
for free. Imposed structure only counts as controlled if it towers over this floor.

A notation note: these pages use $G_\square$, $G_{C_5}$, $G_{C_6}$ for the empty
square, pentagon and hexagon. The 2016 paper separately names its cycle *models*
C1–C4, and those are offset by one (C2 is the square). Be careful quoting either.

---

## 3. The four claims

These describe orbit-resolved CMA with explicit inputs. Two weaker formulations are
excluded because neither is adequate for dataset work. Allocation without orbits gets
asymmetric shapes right only on average. And UDA — the 2017 paper's other algorithm —
lets subgraph counts *emerge* from sampling, so by its own authors' account you cannot
target a clustering level with it. UDA survives in one supporting role: discovering
participation sequences that are known to be buildable.

### [Claim 1 — The degree sequence is exact](claim-1-exact-degrees.md)

Every node ends with exactly the degree you asked for. Not on average. Not
approximately. Element-wise, on every build. The builder asserts the invariant on
every run. The guarantee costs a small sampling bias — derived exactly and measured
at within 3% of uniform per graph on an enumerable sequence — that cancels in
matched pairs.

### [Claim 2 — Subgraph counts are set by the input, not by luck](claim-2-designed-counts.md)

The number of triangles, squares or diamonds you design is the number the generator
builds: realized count = designed count + the by-product floor, with additivity
measured within 2% across every family tested. The floor itself is predictable in
closed form before anything is built. The claim is exact when participation is
*prescribed*; sampled participation can be clipped by the degree budget.

### [Claim 3 — Clustering is known before the graph exists](claim-3-clustering-in-advance.md)

With degrees pinned, the clustering denominator is fixed in advance, and orbit
resolution makes every shape's triangle contribution a known integer. Designed
clustering is a closed-form calculation — `designed_clustering(config)` — and the
realized value lands within ~2% of designed + floor.

### [Claim 4 — Higher-order structure stays free](claim-4-residual-freedom.md)

With degrees and clustering pinned, real freedom remains: which four-node shapes
appear, how shapes are placed, how local clustering varies with degree. That residual
freedom is the experimental variable — and it is an asset only once measured. What you
do not control still moves; the measured confounds are on the claim's page.

---

## 4. The assumptions

Everything above holds only if the inputs are feasible. Six conditions, all checkable:

1. **Parity.** The degrees must sum to an even number. Odd sums strand a stub.
2. **Divisibility.** Shape corners must add up to whole shapes — triangle corners
   divisible by three, and for asymmetric shapes, compatible totals per orbit. Surplus
   corners are broken back down into plain edges.
3. **The degree budget.** A node's shapes must fit inside its degree. Globally:
   average edge-cost of participation ≤ average degree. This is necessary, not
   sufficient — and running it at equality leaves zero slack, which is exactly the
   regime where Claim 2's cap bites hardest. Leave headroom.
4. **Retry, never delete.** A pairing that creates a self-loop or duplicate edge is
   put back and redrawn. Deleting it instead would break Claim 1.
5. **Coverage.** The plain edge $G_0$ almost always belongs in the shape set. Without
   it, some degrees cannot be composed from the available corner sizes.
6. **By-products are unavoidable.** Any generator of this kind produces accidental
   structure; the 2017 paper conjectures this is universal. You cannot remove the
   floor. You can only measure it and design far above it.

---

## 5. How the generator works

Five steps, in plain language. The formal treatment is in
[Sequential Conditional Sampling](../concepts/sequential-conditional-sampling.md) and
[the Clustered Configuration Model](../concepts/ccm.md).

**Input.** A degree sequence. A set of shapes. For each shape, how many copies each
node joins.

**Step 1 — Find the orbits.** Compute each shape's interchangeable positions from its
symmetries. Then split each node's participation across those positions, using an urn
scheme that hits the correct global totals exactly on every draw (the 2017 paper's
multinomial version got them right only on average).

**Step 2 — Check the budget.** Each position costs edges. Verify every node can afford
its allocation; whatever degree is left over becomes plain edges.

**Step 3 — Place.** Decide which nodes host which shape corners. This is a genuine
choice, not bookkeeping: pushing clustered shapes onto hubs changes assortativity and
the clustering-by-degree profile. It can be left to sampling or controlled explicitly.

**Step 4 — Wire.** Put every corner into a bin by shape and position. For each shape
instance, draw its corners from the right bins. On a collision, put the nodes back and
redraw. If the pool deadlocks, reset and rebuild from scratch.

**Verify.** Assert the realized degrees equal the target. Then measure everything
else — clustering, assortativity, subgraph counts — because Claim 4 says the
uncontrolled quantities moved somewhere.

---

## 6. What measurement shows

The implementation is audited against nine acceptance criteria. The scoreboard, in
words:

| Criterion | Status |
|:--|:--|
| P1 Degrees exact per node | ✅ 100% of nodes, all families |
| P2 Edge counts equal | ✅ follows from P1 |
| P3 Signal ≥ 5× the floor | ⚠️ depends on degree choices — computable in advance (see §7) |
| P4 No unasked-for structure above the floor | ✅ measured directly via the four-node census; the one class that moves (the paw) moves by construction |
| P5 Clustering matched by design | ✅ closed-form; realized within ~2% |
| P6 Reproducible from a seed | ✅ |
| P7 Assortativity reported | ✅ reported — and it caught a confound |
| P3-local Participating nodes clear 5× of their *own* floor | ✅ measured — and it disagrees with global P3 in both directions |
| P8 Matched pairs matched per node, not just in totals | ⚠️ one pair leaks: the null vs squares, on long-cycle incidence |

Eight measured findings shape the design advice. Each is a property of the method, not
a footnote:

**The obvious dataset family fails.** The natural family — random null vs square vs
pentagon vs hexagon, all at the same heterogeneous degrees — fails the signal-to-floor
test. Long cycles appear spontaneously in huge numbers once degrees are spread out: the
pentagon and hexagon families land *below* their own floor. The cure is regularity: at
constant degree 4, the same family clears the bar comfortably (triangles 133×, squares
30×, pentagons 11×; hexagons marginal at 4×).

**The floor is a formula.** The driver is the mean excess degree,
$\kappa=\langle k(k-1)\rangle/\langle k\rangle$ — how many onward edges you find after
arriving somewhere along a random edge. The floor for length-$L$ cycles is
$\kappa^L/2L$, and that one expression reproduces every floor measured in this project
to within ~16%. It over-predicts under heavy tails, which is the safe direction: it
never promises a contrast the graph will not deliver. Whether a dataset is possible is
a pre-flight calculation — `predicted_cycle_floor` — not a build-and-see.

**$\kappa$ is the controlling variable, not the distribution's shape.** A power-law
family and a moderate heterogeneous family with the *same* $\kappa$ produce
statistically identical floors at every cycle length. Degree heterogeneity is harmful
exactly because it raises $\kappa$ at the same mean — a measured 3–19× inflation.

**Heavy tails kill cycles but spare cliques.** At power-law exponent 2.5 even the
triangle fails the 5× test (3.6×), and $\kappa$ grows with graph size there, so
scaling up makes it worse, not better. Cliques survive: the $K_5$ floor is exactly
zero in every regime measured, and the $K_4$ floor, though it leaves zero under the
heaviest tail (0.8 at a thousand nodes, 4.6 at four thousand), is outrun by the signal
250-fold. Two caveats travel with that: the clique floor scales with size, so measure
it at the size you intend to ship; and under heavy tails fewer than half the nodes
have the degree to host a $K_4$ at all.

**The degree cap creates an assortativity confound.** Sampled participation is capped
by what each node's degree can afford, which ties placement to degree. Shapes land
preferentially on hubs, and every subgraph family comes out at assortativity ≈ +0.09
against the null's ≈ 0. The cycle families are matched *to each other* — but not to
the null.

**Sampled participation under-delivers at the budget boundary.** At the 2016 cycle
model's parameters the budget constraint is satisfied with equality — zero slack — and
the cap clips participation to 62% of nominal. Prescribing the full amount does not
rescue it; the build then deadlocks because no free stubs remain for pairing.

**Global and local control are different questions.** The square family fails global
P3 at 4.1× — yet 70% of its participating nodes clear 5× against their own local
floor, because the floor lives on hubs and most participants are not hubs. It is
usable for node-level tasks. The mirror image also holds: the triangle family passes
globally at 21× while a third of its hub participants drown locally. Placement cannot
rescue what degree forbids: from degree 4 upward the pentagon floor already exceeds
anything a node's whole degree budget could buy, and aggressive hub-avoidance
backfires — the displaced stubs pile onto the hubs and generate nine times the floor
in *accidental* pentagons.

**Placement direction matters, and it is not the intuitive one.** Giving hubs the
heavy *roles* inside shapes lowers assortativity (−0.12 against +0.03): an asymmetric
shape's edges mostly run from heavy roles to light ones, so ranking roles by degree
sends edges *across* the degree gap. What raises assortativity is concentrating whole
shapes on hubs — which is what the degree cap quietly does.

---

## 7. Rules of thumb for building a dataset

Each rule has a measured reason behind it.

- **Keep mean degree low.** The floor grows exponentially with degree; imposed
  structure grows linearly. This is the single most powerful lever.
- **Keep degrees regular — and compute $\kappa$ first.** The floor is $\kappa^L/2L$
  with $\kappa=\langle k(k-1)\rangle/\langle k\rangle$, computable from the degree
  sequence before anything is built. If the predicted floor drowns your signal, no
  generator can save the design; change the degrees or the shapes.
- **Prefer short cycles and complete shapes.** Triangles and $K_4$s barely occur by
  accident; hexagons occur by the tens of thousands. $K_4$'s floor is zero at
  light-tailed degrees; under a heavy tail it stays small but grows with graph size —
  measure it at the size you intend to use.
- **Use triangles only when clustering is the variable under study.** They are a ~19×
  clustering outlier against every other cycle.
- **Compare cycle families to each other, not to the random null.** The null is not
  assortativity-matched to them (the +0.09 gap). Or build a null matched on
  assortativity.
- **Leave headroom under the degree budget.** At equality, participation clips and
  builds deadlock.
- **Prescribe rather than sample when control matters.** Prescribed participation and
  prescribed orbit splits are exact; sampled ones are exact only in expectation.
- **Judge node-level datasets locally, not globally.** The global 5× test and the
  per-node one disagree in both directions. A family that fails globally can still
  serve node classification on most of its signal-carrying nodes; a family that
  passes globally can still drown its hubs. And placement cannot rescue what degree
  forbids.
- **Report clustering *and* assortativity for every family, every time.** The
  uncontrolled quantities are where the confounds live.

---

## 8. Open questions

- **Sampling bias of the matching algorithm.** The published 2017 paper (§2.3) shows
  the matching algorithm is itself biased and names an unbiased alternative, the
  *refusing algorithm*. The retry bias here has an exact expression and a measured
  value (max per-graph deviation from uniform 2.95% on the audited sequence; see
  [Claim 1](claim-1-exact-degrees.md)) and cancels in matched pairs, but the
  refusing algorithm has not been implemented or compared.
- **The automorphism-cardinality conjecture** from the 2014 paper remains unproven.
  It is the only unproven claim in the corpus.
- **Deliberate hub concentration.** Measurement shows the 2017 paper's assortativity
  effect comes from concentrating whole shapes on hubs, not from role assignment.
  Building that construction deliberately — participation itself concentrated on
  high-degree nodes — has not been done end-to-end.

---

## 9. The evidence

Every number on these pages is reproducible. Four deterministic scripts produce four
JSON result files; the pages cite the numbers, the JSONs hold the exact values.

| Artifact | What it measures |
|:--|:--|
| [`control_audit.py`](evidence/control_audit.py) → [`control_audit_results.json`](evidence/control_audit_results.json) | Degrees, additivity, clustering, assortativity, orbit splits, the four-node census |
| [`audit_section_h.py`](evidence/audit_section_h.py) → [`audit_section_h_results.json`](evidence/audit_section_h_results.json) | Per-node (local) signal-to-floor, matched-pair leakage, placement experiments |
| [`audit_section_i.py`](evidence/audit_section_i.py) → [`audit_section_i_results.json`](evidence/audit_section_i_results.json) | The $\kappa^L/2L$ floor formula, heavy-tail regimes, clique floors |
| [`uniformity_bias.py`](evidence/uniformity_bias.py) → [`uniformity_bias_results.json`](evidence/uniformity_bias_results.json) | The sampler's exact deviation from uniform, closed form and Monte Carlo |

Code entry points: `ConfigModelGraph.from_config`, `SubgraphSequence`,
`designed_clustering`, `predicted_cycle_floor`, `mean_excess_degree` — see the
[metrics API](../api/metrics.md) and the
[configuration model API](../api/configuration-model.md).
