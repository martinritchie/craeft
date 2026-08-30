# Sequential Conditional Sampling for Orbit Assignment

## The Problem

Consider a non-vertex-transitive subgraph like the diamond (G7). Each diamond
instance needs four nodes, but the four positions are not equivalent: two are
**hubs** (connected to all three others) and two are **leaves** (connected to
only two). These structurally distinct roles are the vertex **orbits**.

We know how many subgraph instances each node participates in (the participation
sequence), but not **which orbit role** it fills in each instance. That is the
assignment problem.

### Notation

The subgraph $H$ has $k$ vertex orbits under its automorphism group. Each
orbit $o$ has:

- **size** $\sigma_o$: how many vertices in $H$ belong to orbit $o$
- **degree** $\delta_o$: the degree of those vertices within $H$ (all vertices
  in the same orbit share the same degree — automorphisms preserve adjacency)

Each node $i$ in the graph has:

- $s_i$: its **participation count** — how many instances of $H$ it appears in
- $d_i$: its **degree budget** — the stubs still available after accounting for
  other subgraph types

### The assignment matrix

We need a matrix $C \in \mathbb{Z}_{\geq 0}^{n \times k}$ where $C_{io}$ is
the number of times node $i$ fills a position in orbit $o$.

For example, with a diamond subgraph and four graph nodes:

$$C = \begin{pmatrix} 2 & 1 \\ 1 & 1 \\ 1 & 1 \\ 0 & 1 \end{pmatrix}$$

This says node 0 fills the hub role twice and the leaf role once (three
participations total), node 3 fills the leaf role once, and so on.

### Constraints

Let $M$ be the total number of subgraph instances:

$$M = \frac{\sum_{i=1}^{n} s_i}{\sum_{o=0}^{k-1} \sigma_o} = \frac{\sum_{i=1}^{n} s_i}{|V(H)|}$$

The assignment must satisfy three constraints:

**1. Row sums** — each node's orbit counts must add up to its participation total:

$$\sum_{o=0}^{k-1} C_{io} = s_i \quad \forall\, i$$

**2. Column sums** — across all nodes, each orbit must be filled the right
number of times. If there are $M$ diamond instances and each needs 2 hubs,
there must be $2M$ hub assignments total:

$$\sum_{i=1}^{n} C_{io} = M \cdot \sigma_o \quad \forall\, o$$

**3. Degree budget** — each orbit role costs a certain number of stubs (a hub
costs 3, a leaf costs 2). The total stub cost for a node can't exceed its
degree:

$$\sum_{o=0}^{k-1} C_{io} \cdot \delta_o \leq d_i \quad \forall\, i$$

The set of all non-negative integer matrices with prescribed row and column sums
is the **transportation polytope** (Hitchcock, 1941). The degree budget carves
out a subset of this polytope.

---

## Why Sequential?

The naive approach is to let each node independently choose its orbit split,
then reject the whole assignment if the global totals are wrong. This fails in
practice: the column-sum constraint (constraint 2 above) couples all $n$ nodes
together, and the probability that independent draws produce the exact right
column totals shrinks exponentially with $n$.

Sequential conditional sampling avoids this by processing nodes one at a time,
drawing each node's split from the **conditional distribution given what
remains**. This guarantees the column sums are met exactly, by construction.

### The card-dealing analogy

Imagine an urn containing coloured balls — one colour per orbit. The urn starts
with exactly $M \cdot \sigma_o$ balls of each colour $o$. Node $i$ draws $s_i$
balls from the urn. Once drawn, those balls are gone; the next node draws from
whatever is left.

After all $n$ nodes have drawn, the urn is empty and each colour's total is
exactly right by construction. The only additional complication is the degree
budget, which limits how many "expensive" balls (high-degree orbits) a node can
afford to draw.

---

## The Two-Orbit Case

Most non-transitive subgraphs in practice have exactly two orbits ($k = 2$):
the diamond (G7), house (G17), and bowtie (G14). With only two orbits, the
problem has a clean closed-form solution.

### Reducing to one variable

With two orbits, once you choose $C_{i0}$ (how many hub roles), the leaf count
is fixed: $C_{i1} = s_i - C_{i0}$. So define $x_i \coloneqq C_{i0}$ and the
entire problem becomes: find integers $x_1, \ldots, x_n$ such that

$$\sum_{i=1}^{n} x_i = T_0 \coloneqq M \cdot \sigma_0 \qquad \text{(global target for orbit 0)}$$

$$\ell_i \leq x_i \leq u_i \quad \forall\, i \qquad \text{(per-node bounds)}$$

### Where the bounds come from

Each $x_i$ is constrained by three independent considerations:

**Non-negativity** — both orbit counts must be non-negative. Since $C_{i0} = x_i$
and $C_{i1} = s_i - x_i$, this simply gives $0 \leq x_i \leq s_i$.

**Degree budget** — filling a hub role costs $\delta_0$ stubs; a leaf costs
$\delta_1$. The total cost is $x_i \cdot \delta_0 + (s_i - x_i) \cdot \delta_1$.
This must not exceed $d_i$.

Rearranging (assuming $\delta_0 > \delta_1$, i.e. orbit 0 is the expensive one):

$$x_i \cdot (\delta_0 - \delta_1) \leq d_i - s_i \cdot \delta_1$$

$$x_i \leq \left\lfloor \frac{d_i - s_i \cdot \delta_1}{\delta_0 - \delta_1} \right\rfloor$$

*Intuition:* start by assuming all participations are cheap (leaves). That costs
$s_i \cdot \delta_1$ stubs. The remaining budget $d_i - s_i \cdot \delta_1$ is
the headroom for "upgrading" leaves to hubs, each upgrade costing an extra
$\delta_0 - \delta_1$ stubs.

If $\delta_0 = \delta_1$ (uniform cardinality), the degree constraint doesn't
restrict the split at all — the problem degenerates to the vertex-transitive
case.

**Remaining global demand** — the urn only has $R_0$ orbit-0 balls and $R_1$
orbit-1 balls left. So $x_i \leq R_0$ (can't take more hub slots than remain).
And if there aren't enough leaf slots left to absorb the remainder, some
participations *must* go to hubs: $x_i \geq s_i - R_1$.

**Combined bounds:**

$$\ell_i = \max\!\Big(0,\; s_i - R_1^{(i)}\Big)$$

$$u_i = \min\!\left(s_i,\; R_0^{(i)},\; \left\lfloor \frac{d_i - s_i \cdot \delta_1}{\delta_0 - \delta_1} \right\rfloor\right)$$

where $R_o^{(i)}$ is the **remaining demand** for orbit $o$ at the point when node
$i$ is processed.

### The algorithm

Walk through nodes one at a time, maintaining how many orbit-0 and orbit-1
slots remain in the urn.

---

**Initialise:** $R_0 \leftarrow M \cdot \sigma_0$, $\;R_1 \leftarrow M \cdot \sigma_1$

**For** $i = 1, \ldots, n$:

1. **Compute feasible bounds** on $x_i$ (how many hub roles node $i$ takes):

$$\ell_i = \max\!\Big(0, \; s_i - R_1\Big)$$

$$u_i = \min\!\left(s_i, \; R_0, \; \left\lfloor \frac{d_i - s_i \cdot \delta_1}{\delta_0 - \delta_1} \right\rfloor\right)$$

2. **Feasibility check:** if $\ell_i > u_i$, no valid split exists for this
   node — the participation sequence is incompatible with the degree budget.
   Raise and retry with a fresh participation sequence.

3. **Sample** $x_i$ uniformly from $\{\ell_i, \ldots, u_i\}$.

4. **Record:** $C_{i0} = x_i$, $\;C_{i1} = s_i - x_i$.

5. **Deplete the urn:** $R_0 \leftarrow R_0 - x_i$, $\;R_1 \leftarrow R_1 - (s_i - x_i)$.

---

**Time complexity:** $O(n)$ — one pass, constant work per node.

**Space complexity:** $O(n)$ for the output matrix (or $O(1)$ auxiliary).

### A worked example

Consider the diamond (G7):

```
    0
   / \
  1   2      Orbit 0 (hubs): nodes {0, 3} — degree 3 within the diamond
   \ /       Orbit 1 (leaves): nodes {1, 2} — degree 2 within the diamond
    3
```

- Orbit 0: $\sigma_0 = 2$ vertices, $\delta_0 = 3$ stubs each
- Orbit 1: $\sigma_1 = 2$ vertices, $\delta_1 = 2$ stubs each

Suppose $n = 4$ graph nodes, degree budget $d = [8, 6, 5, 5]$, and
participation sequence $s = [3, 2, 2, 1]$.

Total participations: $\sum s_i = 8$, subgraph size $|V(H)| = 4$, so $M = 2$
diamond instances.

Global targets: $T_0 = 2 \times 2 = 4$ hub slots, $\;T_1 = 2 \times 2 = 4$
leaf slots.

The urn starts with 4 red balls (hub) and 4 blue balls (leaf).

---

**Node 0** draws $s_0 = 3$ balls. Budget $d_0 = 8$. Urn: $R_0 = 4$, $R_1 = 4$.

Lower bound: $\ell_0 = \max(0, \; 3 - 4) = 0$ — the urn has enough leaf balls
to cover all 3 draws, so no hubs are forced.

Upper bound: $u_0 = \min(3, \; 4, \; \lfloor(8 - 6) / 1\rfloor) = \min(3, 4, 2) = 2$ — the degree budget is the binding constraint. All-leaf costs $3 \times 2 = 6$ stubs, leaving headroom for 2 upgrades to hub.

Sample $x_0 \in \{0, 1, 2\}$. Suppose $x_0 = 2$ (2 hubs, 1 leaf).

Urn after: $R_0 = 4 - 2 = 2$ red, $\;R_1 = 4 - 1 = 3$ blue.

---

**Node 1** draws $s_1 = 2$ balls. Budget $d_1 = 6$. Urn: $R_0 = 2$, $R_1 = 3$.

Lower bound: $\ell_1 = \max(0, \; 2 - 3) = 0$.

Upper bound: $u_1 = \min(2, \; 2, \; \lfloor(6 - 4) / 1\rfloor) = 2$.

Sample $x_1 \in \{0, 1, 2\}$. Suppose $x_1 = 1$ (1 hub, 1 leaf).

Urn after: $R_0 = 1$, $R_1 = 2$.

---

**Node 2** draws $s_2 = 2$ balls. Budget $d_2 = 5$. Urn: $R_0 = 1$, $R_1 = 2$.

Lower bound: $\ell_2 = \max(0, \; 2 - 2) = 0$.

Upper bound: $u_2 = \min(2, \; 1, \; \lfloor(5 - 4) / 1\rfloor) = 1$ — the
urn only has 1 hub ball left, so $R_0$ is the binding constraint.

Sample $x_2 \in \{0, 1\}$. Suppose $x_2 = 1$ (1 hub, 1 leaf).

Urn after: $R_0 = 0$, $R_1 = 1$.

---

**Node 3** draws $s_3 = 1$ ball. Budget $d_3 = 5$. Urn: $R_0 = 0$, $R_1 = 1$.

Lower bound: $\ell_3 = \max(0, \; 1 - 1) = 0$.

Upper bound: $u_3 = \min(1, \; 0, \; 3) = 0$ — no hub balls remain.

Forced: $x_3 = 0$ (1 leaf). No choice at all.

Urn after: $R_0 = 0$, $R_1 = 0$. Empty.

---

**Result:**

| Node | $s_i$ | $C_{i0}$ (hub) | $C_{i1}$ (leaf) | Stub cost | Budget $d_i$ | Remaining |
|------|-------|-----------------|-----------------|-----------|---------|-----------|
| 0 | 3 | 2 | 1 | $2(3) + 1(2) = 8$ | 8 | 0 |
| 1 | 2 | 1 | 1 | $1(3) + 1(2) = 5$ | 6 | 1 |
| 2 | 2 | 1 | 1 | $1(3) + 1(2) = 5$ | 5 | 0 |
| 3 | 1 | 0 | 1 | $0(3) + 1(2) = 2$ | 5 | 3 |

Column sums: $C_{\cdot,0} = 4 = T_0$ ✓, $\;C_{\cdot,1} = 4 = T_1$ ✓. The urn
is empty and the global orbit proportions are exactly satisfied.

Notice how later nodes become increasingly constrained — not because of their
own properties, but because earlier nodes consumed from the shared pool. Node 3
had no choice at all. This is the sequential conditioning at work.

---

## The General $k$-Orbit Case

For subgraphs with $k > 2$ orbits, the reduction to a single variable no longer
applies. At each node we must sample a $k$-vector — how many of each orbit
role to assign.

In practice this is rare. Subgraphs in the graphlet registry have at most 6
nodes, yielding $k \leq 3$ orbits. But the algorithm generalises cleanly.

### Feasible set at node $i$

At each node, the set of valid orbit assignments is a small **local polytope**:

$$\mathcal{F}_i = \left\{ \mathbf{c} \in \mathbb{Z}_{\geq 0}^{k} \;\middle|\; \sum_{o} c_o = s_i, \;\; c_o \leq R_o^{(i)}, \;\; \sum_{o} c_o \cdot \delta_o \leq d_i \right\}$$

In words: pick non-negative integers $c_0, \ldots, c_{k-1}$ that add up to
$s_i$ (right number of participations), don't exceed what the urn has left for
any orbit ($c_o \leq R_o$), and don't blow the degree budget.

### Sampling strategy by orbit count

**$k = 1$ (vertex-transitive):** Nothing to do. All participations go to the
single orbit: $C_{i0} = s_i$.

**$k = 2$:** The two-orbit algorithm above. $O(1)$ per node.

**$k = 3$:** Fix $c_0$ at some feasible value, then the remaining two orbits
form a $k = 2$ sub-problem. Iterate over all feasible $c_0$ values and sample
uniformly among valid $(c_0, c_1, c_2)$ triples. Cost is $O(s_i)$ per node —
$s_i$ is typically small.

**General $k$:** Enumerate lattice points of $\mathcal{F}_i$ and sample
uniformly. For $k \leq 3$ and moderate $s_i$, the number of lattice points is
small. For larger $k$, Markov chain methods (Diaconis & Gangolli, 1995) apply,
but this is unlikely to be needed for graphlet-scale subgraphs.

### The algorithm (general)

The structure is identical to the two-orbit case — walk through nodes, draw
from the urn, deplete — but the per-node draw is a $k$-vector instead of a
scalar.

---

**Initialise:** $R_o \leftarrow M \cdot \sigma_o$ for each orbit $o$.

**For** $i = 1, \ldots, n$:

1. **Compute** $\mathcal{F}_i$ given current remaining demands $R_0, \ldots, R_{k-1}$
   and degree budget $d_i$.

2. **Feasibility check:** if $\mathcal{F}_i = \emptyset$, raise and retry.

3. **Sample** $\mathbf{c}^{(i)}$ uniformly from $\mathcal{F}_i$.

4. **Record:** $C_{io} = c_o^{(i)}$ for each orbit $o$.

5. **Deplete the urn:** $R_o \leftarrow R_o - c_o^{(i)}$ for each orbit $o$.

---

**Time complexity:** $O(n \cdot f(k, s_{\max}))$ where $f$ is the cost of
enumerating $\mathcal{F}_i$. For $k = 2$ this is $O(n)$.

---

## Processing Order

The algorithm produces a valid assignment regardless of which node goes first.
But the order affects practical performance.

### Fail fast: process high-participation nodes first

A node with many participations ($s_i$ large) has the tightest constraints — it
needs lots of stubs and must distribute them across orbits. If the participation
sequence is incompatible with the degree budget, this is the node most likely to
expose it.

Processing nodes in descending $s_i$ order detects infeasibility early, avoiding
wasted work on later nodes.

### Sampling bias: early nodes get more freedom

Uniform sampling at each step does **not** yield a globally uniform distribution
over the transportation polytope. Early nodes face a full urn and wide bounds;
late nodes face a depleted urn and narrow bounds (or are fully determined, like
node 3 in the worked example).

This is the same phenomenon as in the standard configuration model: the first
stubs paired have free choice, the last stubs are forced. It doesn't affect
correctness — the theoretical properties of the CMA come from the **outer
rejection loop** (discard invalid graphs), not from the internal sampling being
perfectly uniform.

If exact polytope-uniformity were needed, it can be recovered with importance
weights:

$$w(C) = \prod_{i=1}^{n} |\mathcal{F}_i|$$

where $|\mathcal{F}_i|$ is the feasible range size at the time node $i$ was
processed. Resampling with probability $\propto 1/w$ corrects the bias. In
practice this is not needed for CMA.

---

## Prescribing the Split Directly

Sampling gets the *global* orbit totals exactly right ($M\sigma_o$ on the
nose) but *which* node lands in which orbit is a random draw. For an
asymmetric subgraph like the diamond, orbits carry different triangle
counts (hubs sit in more triangles than leaves), so a node's designed
local clustering is only correct **in expectation** — two nodes with
identical participation $s_i$ can walk away with different orbit splits,
and therefore different per-node designed triangle counts, purely because
of draw order and the luck of the urn. Global $C$ is invariant across
re-splits; the per-node $c_i$ profile is not.

That's fine when only the aggregate matters. It's a problem when the
*placement* of structure is itself the thing under study (e.g. a
benchmark that varies where clustering sits, or holds the $c(k)$ profile
fixed while changing something else). For that, `SubgraphSequence`
accepts a prescribed decomposition:

```python
SubgraphSequence(
    subgraph=diamond,
    orbit_counts={0: hub_counts, 1: leaf_counts},
)
```

When `orbit_counts` is set, `_split_by_orbit` returns it unchanged —
sampling is bypassed entirely, including the vertex-transitive
early-return (a single-orbit subgraph can still be prescribed; it just
has one key). The participation sequence and RNG passed to
`_split_by_orbit` are ignored in this case.

`distribution` is omitted above, and that is deliberate: a fully
prescribed sequence has no participations left to draw, so
`from_config` never calls `sample` for it. Supply `distribution` *or*
`orbit_counts`; a sequence with neither is rejected on construction.
(Passing both is allowed but the distribution is inert — the
prescription wins.) Because the prescription also pins the instance
count exactly, `SubgraphSequence.num_instances` returns that integer
$M$ rather than `None`, and the designed-clustering metrics use it in
place of $n \cdot \mathbb{E}[s] / |V(H)|$ — making the designed figures
*exact* for a prescribed sequence rather than correct-in-expectation.

### Validation happens at two layers

The split is between what a sequence can know about *itself* and what
only the surrounding config knows.

**On construction (`__post_init__`)** — self-consistency, so a
malformed prescription fails at the point it is written:

1. **Keys** must equal the orbit labels exactly.
2. **Arrays** must all be the same length and non-negative.
3. **Totals** must satisfy $\sum_i C_{io} = M \cdot \sigma_o$ for a
   *common* integer $M$ across every orbit — the same column-sum
   invariant the urn sampler enforces implicitly through depletion,
   made explicit and checked up front. A prescription that would form,
   say, 2 hub-instances but only 1 leaf-instance is rejected.

Note what is *absent*: the array length is not checked against $n$,
because a `SubgraphSequence` has no idea which graph it will be used
with. Nor is the degree budget, for the same reason.

**Pre-flight in `from_config`** — everything needing the config, run
**once, before the retry loop**:

4. **Length** of every prescribed array must equal $n$ → `ValueError`.
5. **Degree budget**: the prescribed sequences' combined stub cost
   $\sum_o C_{io} \delta_o$ must fit within $k_i$ for every node →
   `AllocationError` naming the offending nodes.

Placement is the whole point of that second layer. The retry loop
exists for *stochastic* dead ends — a draw that happened to be
infeasible, where a fresh draw might not be. A prescription is
deterministic: if it doesn't fit, it doesn't fit on attempt 200
either. Left inside the loop, such a failure spends every retry
rediscovering the same fact and then reports
`RuntimeError: Failed to generate graph after N retries`, with the
actual diagnosis demoted to `__cause__`. Hoisting it out keeps the
error legible and the failure instant.

Row sums (does $\sum_o C_{io}$ match a real participation count for node
$i$?) are still not checked anywhere — `orbit_counts` fully replaces the
participation sequence, so there is nothing to compare against.

When prescribed and sampled sequences are mixed, the pre-flight budget
check covers the prescribed ones only: passing it is necessary but not
sufficient, and `allocate_subgraphs` remains the authoritative check
inside the loop. Failing it is conclusive — the prescription cannot fit
whatever is sampled alongside it. Either way a bad prescription is
surfaced, never silently repaired.

### Building a prescription: two helpers

Hand-building `orbit_counts` is impractical for anything beyond a toy
example, so two constructors cover the common cases. Both take a
participation sequence and return a dictionary in the same shape
`orbit_counts` expects — feed the output straight back in.

**`split_deterministic(sequence, orbits)`** — the "fair" split. Uses
largest-remainder (Hamilton) apportionment to give each node its
proportional share of each orbit, so nodes with equal participation
get equal orbit counts wherever divisibility allows. Unlike the urn
sampler, this has no processing-order dependence: the split for node
$i$ depends only on $s_i$, not on what earlier nodes happened to draw.

**`split_by_degree_rank(sequence, degrees, orbits)`** — the "push
clustered subgraphs onto hubs" construction. Ranks orbits by
within-subgraph degree $\delta_o$ (hub roles first) and nodes by
graph degree (highest first), then greedily fills each orbit's
target from the highest-remaining-degree nodes with capacity left. This
is the 2017 paper's §3.3 construction, which "opted to push the
clustered subgraphs onto the higher-degree nodes to accentuate the
effect of clustering" and produced measurably more assortative graphs
(2017 Fig. 8). Doing it deliberately, rather than as a side effect of
greedy allocation order, makes that assortativity shift (ticket 005) a
controlled variable instead of an incidental one.

Both helpers keep the vertex-transitive early return: a single-orbit
subgraph is passed through unchanged, since there's nothing to split.

---

## Feasibility Guarantee

Before running the full loop, you can cheaply check whether a valid assignment
exists at all (for the two-orbit case). Compute every node's bounds and check:

$$\sum_{i=1}^{n} u_i \geq T_0 \qquad \text{and} \qquad \sum_{i=1}^{n} \ell_i \leq T_0$$

*Intuition:* the first condition says "if every node took as many hub roles as
it possibly could, is that enough?" The second says "if every node took as few
hub roles as it could get away with, would that already overshoot?"

For $k = 2$, these conditions are both necessary **and** sufficient — a
consequence of the max-flow min-cut theorem on the trivial bipartite graph. If
they hold, the sequential algorithm is guaranteed to complete without hitting
infeasibility at any step.

For $k > 2$, these per-orbit conditions are necessary but not always sufficient.
Full feasibility requires checking the transportation polytope intersected with
the degree constraints, which is itself a linear program.

### Infeasibility is expected

In the CMA pipeline, participation sequences are sampled independently of the
degree sequence. Some draws will be incompatible — a node might be told to
participate 5 times in diamonds but only have degree 6, which can't even cover
5 leaf roles ($5 \times 2 = 10 > 6$).

This is by design. The retry loop in `ConfigModelGraph.from_config` handles it:
resample participation sequences and try again. The feasibility check above lets
you reject bad draws immediately, before entering the sequential loop.

---

## Connection to Established Work

The algorithm here isn't novel — it combines well-studied building blocks.

**Contingency table sampling.** The assignment matrix $C$ is a contingency table
(non-negative integer matrix with fixed row and column sums). Sampling such
tables is a classical problem. Patefield (1981) gives an efficient sequential
conditional algorithm — our algorithm is a direct specialisation of his, with
the addition of the per-row degree constraint.

**Transportation problem.** Finding *any* feasible $C$ (not a random one) is a
transportation problem — a special case of linear programming with efficient
dedicated solvers. The feasibility check in the previous section is the dual
of this.

**Configuration model.** The standard configuration model (Bollobás, 1980;
Molloy & Reed, 1995) pairs stubs uniformly at random to realise a degree
sequence. The CMA (Ritchie et al., 2015) adds a subgraph layer on top: nodes
are first assigned to subgraph instances, then stubs within each instance are
connected. The sequential conditional sampler slots into this pipeline between
participation sequence sampling and subgraph connection.

| Concept | Reference |
|---|---|
| Transportation polytope | Hitchcock (1941), Koopmans (1949) |
| Contingency table sampling | Patefield (1981), *Applied Statistics* 30(1), 91-97 |
| Sequential conditional method | Chen, Dempster & Liu (1994) |
| MCMC on integer matrices | Diaconis & Gangolli (1995) |
| Configuration model | Bollobás (1980), Molloy & Reed (1995) |
| CMA algorithm | Ritchie, Berthouze & Kiss (2015), *J. Complex Networks* 5(1) |
