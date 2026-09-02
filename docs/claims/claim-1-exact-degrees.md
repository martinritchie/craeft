# Claim 1 — The degree sequence is exact

**The number of edges (stubs) originating from each node is exact.**

## What is held

| | |
|:--|:--|
| **Exact** | Realized degree = target degree, per node, every build. Asserted at build time: a violation raises rather than returning a graph. |
| **Approximate** | Uniformity of the sampler over the simple graphs with that degree sequence. On the audited sequence no graph's probability sits more than 3% from uniform, and the typical deviation falls like $1/\sqrt{m}$ with the number of edges. Both sides of a matched pair come from the same sampler, so the bias is common-mode and cancels in the comparison. |
| **Not controlled** | Everything else — which is the point of [Claim 4](claim-4-residual-freedom.md). |

## How it is held

The sampler is the pairing step of the matching algorithm. Give node $u$ a pool
of $d_u$ stubs, then repeat until the pool is empty:

1. Draw two stubs uniformly at random from the pool.
2. If they sit on the same node, or their nodes are already joined, put both
   back and draw again.
3. Otherwise record the edge and remove both stubs.

If stubs remain but no pair can pass step 2, abandon the build and start again
from the full pool.

Two of those rules carry the guarantee. Step 2 puts a colliding pair back rather
than deleting it: dropping it would silently lose an edge, and the "matched"
degree sequences would no longer be matched. The restart rule refuses to return
a graph from a dead end. Both rules matter below. The first is where the
sampling bias comes from; the second is what the bias is conditioned on.

## What the guarantee costs

Retrying collisions makes the sampler non-uniform over the simple graphs with the
target degree sequence. The alternative, restart-from-scratch rejection — the
*repeated configuration model* of the literature — is exactly uniform, but on the
audited sequence it discards 70% of draws. Retrying discards 31%, and pays for the
difference with a per-graph bias of at most ±2.95% on the five-edge audit
sequence. The bias has a known form: to leading order it favours graphs whose
edges join high-degree nodes, and its typical size falls like $1/\sqrt{m}$ with
the number of edges, though its worst case does not. A benchmark compares two
families built by the same sampler, so the bias is present in both and the
comparison sees only their difference. The rest of this section measures that
shared bias: how large it is, and which graphs it favours.

### Origin of the bias

The analysis is of the plain-edge pairing, the code path that wires leftover
stubs into single edges. The shape bins pair hyperstubs into whole subgraphs
under the same retry rule, and their sampling bias is not analysed here.

The pairing reduces to a chain whose state is the set of edges formed so far. An
edge is *committed* at step 3; a redraw at step 2 leaves the state unchanged, a
self-transition that drops out of the analysis. Take first the configuration
model itself, which is the same procedure without step 2: every pair of distinct
stubs is admissible, self-loops and duplicates included. With $T$ stubs
remaining there are $A = T(T-1)/2$ such pairs, and because step 1 draws a
uniform pair, edge $\{u, v\}$ is committed with probability $s_u s_v / A$, where
$s_u$ is node $u$'s residual stub count.

Now fix a simple graph $g$ with $m$ edges and target degrees $d$. The process
reaches $g$ by committing its edges in some order $\pi$, and the probability of one
such path is the product of the per-commit probabilities,

$$\prod_{t=1}^{m} \frac{s_{u_t} s_{v_t}}{A_t},$$

where $\{u_t, v_t\}$ is the $t$-th edge of the ordering and $A_t$ is the pair
count after the first $t-1$ commits.

The numerator of this product depends on neither the ordering nor the graph. Node
$u$'s residual count starts at $d_u$ and falls by one at each of its $d_u$ edges,
so across any ordering it takes each value $d_u, d_u - 1, \ldots, 1$ exactly once.
The numerator therefore collapses to $\prod_u d_u!$, and all that distinguishes
one path from another is its normalisers.

Nor, here, do the normalisers distinguish paths. $A_t$ depends only on how many
stubs remain, $T_t = 2m - 2(t-1)$, not on which edges exist, so every ordering
of $g$'s edges has the same product of them. Summing over the $m!$ orderings,

$$\sum_{\pi} \prod_{t=1}^{m} \frac{1}{A_t} \;=\; \frac{m!\,2^m}{(2m)!} \;=\; \frac{1}{(2m-1)!!},$$

the last step because $(2m)!$ splits into its even factors $2^m\,m!$ and its
odd factors $(2m-1)!!$. That double factorial counts the ways to pair $2m$
stubs, so the right-hand side is the probability of one particular pairing under
uniform matching. Writing $P_{\mathrm{CM}}$ for the configuration model's law,

\begin{equation}
P_{\mathrm{CM}}(g) \;=\; \frac{\prod_u d_u!}{(2m-1)!!}
\label{eq:cm}
\end{equation}

for every simple graph alike. That is van der Hofstad's equation (7.5.2)
(reference below), the configuration model's law restricted to simple graphs. It
also explains why the repeated configuration model is exactly uniform: a uniform
pairing conditioned on being simple is still uniform over the simple graphs.

Now restore step 2. A drawn pair is put back when its stubs sit on one node or
its nodes are already joined, so conditional on leaving a state the committed
edge is $\{u, v\}$ with probability $s_u s_v / W$, where

\begin{equation}
W \;=\; \frac{T(T-1)}{2} \;-\; \sum_u \frac{s_u(s_u-1)}{2} \;-\; \sum_{(a,b) \in E} s_a s_b.
\label{eq:W}
\end{equation}

The first term is $A$. The second removes pairs on the same node, which would
form a self-loop; the third removes pairs that would duplicate an edge already
present. Those two subtractions are the *exclusion terms*, and they are the only
place the sampler differs from the configuration model.

Everything above survives except the last step. The path product has the same
numerator, which collapses to $\prod_u d_u!$ exactly as before, since that
argument never looked at the normaliser. But $W_t$ now depends on which edges
have been committed, not only on how many, so different orderings of $g$'s edges
have different products of normalisers and the sum over orderings no longer
collapses. The probability that the process completes at $g$ is

\begin{equation}
P(g) \;=\; \Big(\prod_u d_u!\Big) \sum_{\pi} \prod_{t=1}^{m} \frac{1}{W_t(\pi)}.
\label{eq:law}
\end{equation}

So the sampler's law is the configuration model's with a single substitution —
the constant $(2m-1)!!$ replaced by an ordering- and graph-dependent product of
normalisers — and the bias lives entirely in the two exclusion terms of
$\eqref{eq:W}$. The retrying sampler renormalises at every step over a pool that
depends on the state; the repeated configuration model conditions once, on the
whole pairing being simple, and that is why one is biased and the other is not.

Two checks on the closed form, both on the audit sequence of the measurement
section. Against the chain: the ordering sum in $\eqref{eq:law}$ and the exact
evaluation of the chain agree on every graph. Against the literature: evaluating
$\eqref{eq:law}$ with $A_t$ in place of $W_t$ must return the book's value
$\eqref{eq:cm}$, $\prod_u d_u!/(2m-1)!! = 48/945 = 16/315$ on that sequence,
and it does on all six graphs. The second check is stronger than it looks. The
book reaches $\prod_u d_u!$ by an unrelated route — as the number of ways to
permute the half-edges at each vertex without changing the graph — so the
collapse of the numerator rests on two independent derivations of one constant.

### Quantifying the bias

The aim is the size of the bias and which graphs it falls on, as a function of
the degree sequence and of $m$. What there is to go on is $\eqref{eq:cm}$ and
$\eqref{eq:law}$: the same expression without and with the exclusion terms, the
first of them uniform. So write $\eqref{eq:law}$ as $\eqref{eq:cm}$ times a
correction, and ask how the correction behaves when the excluded weight is a
small share of the whole. It is, once the graph is large: of the roughly
$T_t^2/2$ candidate stub pairs at a step, only those on a single node or across
an existing edge are forbidden.

**The correction factor.** Write $W_t = A_t - X_t$, with $A_t$ the
configuration model's pair count and $X_t$ the excluded weight, the two
subtractions in $\eqref{eq:W}$. Then $\eqref{eq:law}$ factors:

\begin{equation}
P(g) \;=\; P_{\mathrm{CM}}(g)\; R(g), \qquad
R(g) \;=\; \mathbb{E}_\pi \prod_{t=1}^{m} \frac{1}{1 - X_t(\pi)/A_t},
\label{eq:ratio}
\end{equation}

the expectation over a uniformly random ordering of $g$'s edges. All of the
bias is the variation of $R$ across graphs; a constant $R$ would leave the
sampler uniform. Keeping the leading term in $X_t/A_t$ gives
$\log R(g) = B(g)$ with $B(g) = \mathbb{E}_\pi \sum_t X_t/A_t$. That is what
"leading order" means on this page: the sense of a Taylor expansion, unrelated
to the "higher-order structure" of [Claim 4](claim-4-residual-freedom.md).

**The leading-order formula.** $B$ splits along the two exclusion terms of
$\eqref{eq:W}$, and only the duplicate term sees the graph:

\begin{equation}
B(g) \;=\; C_m(d) \;+\; \Phi_m\, S(g), \qquad
S(g) \;=\; \sum_{(a,b)\in g} (d_a-1)(d_b-1),
\label{eq:leading}
\end{equation}

with $C_m(d)$ independent of $g$ and

$$\Phi_m \;=\; \sum_{j=1}^{m-2} \frac{j\,(m-1-j)}{(2j+1)\,m(m-1)(m-2)}
\;\sim\; \frac{1}{4m}.$$

The self-loop term is $C_m(d)$ because it depends only on when each node's
edges are committed, and under a random ordering the positions of node $u$'s
$d_u$ edges are a uniform $d_u$-subset of the $m$ steps whatever the graph.
The duplicate term is where the graph enters: edge $(a,b)$, once committed,
excludes weight $s_a s_b/A_t$ at every later step, and averaging over where it
and $a$'s and $b$'s other edges fall leaves $(d_a-1)(d_b-1)$ times a
coefficient in $m$ alone. That average is a hypergeometric count, carried out
in the evidence script.

**What it says.** To leading order the sampler weights each graph by
$\exp(\Phi_m S(g))$. It favours graphs whose edges join nodes of high excess
degree: a degree-correlation bias. On the audit sequence that picks out the
three under-sampled graphs and predicts their deviation, as the next section
shows. In the large-$m$ limit it separates two cases.

**Regular sequences.** Every graph has $S = m(d-1)^2$, so the leading-order
bias vanishes identically. Asymptotic uniformity of exactly this sampler is a
theorem there: Steger and Wormald (1999), extended by Kim and Vu (2003) to
$d = O(n^{1/3-\epsilon})$.

**Irregular sequences.** The relative deviation of a graph is
$\Phi_m\,(S(g) - \bar S)$. Across uniform graphs $S$ has spread of order
$\sqrt{m}$, so the typical deviation and the total variation distance fall like
$1/\sqrt{m}$. The extreme graphs, the most and least degree-correlated wirings,
differ in $S$ by order $m$, so the worst-case deviation stays of order one.
Bayati, Kim and Saberi (2010) remove that term by weighting each candidate pair
by $(1 - d_a d_b/4m)$ and prove near-uniformity for $d_{\max} = O(m^{1/4-\epsilon})$.
The sampler here uses the unweighted pair, so the bias is as stated.

The $1/\sqrt{m}$ law is checked by tiling the audit sequence $k$ times, drawing
400 exactly uniform simple graphs by rejection at each size, and evaluating
$\Phi_m\,(S - \bar S)$.

| $m$ | $4m\,\Phi_m$ | Typical deviation (std) | Largest of 400 | $\sqrt{m}\,\times$ typical |
|--:|--:|--:|--:|--:|
| 5 | 0.74 | 1.85% | 1.95% | 0.041 |
| 10 | 0.83 | 2.06% | 4.41% | 0.065 |
| 20 | 0.89 | 1.76% | 5.08% | 0.079 |
| 40 | 0.93 | 1.26% | 5.21% | 0.080 |
| 80 | 0.96 | 0.96% | 3.06% | 0.085 |
| 160 | 0.97 | 0.65% | 2.46% | 0.082 |
| 320 | 0.98 | 0.52% | 1.94% | 0.093 |

The scaled column settles near 0.08 from $m = 20$. The largest-of-400 column is
a sample maximum, not the worst case, which the argument above puts at order
one. At $m = 5$ the leading-order figure understates the exact 2.95% because the
excluded weight is not small there. No exact evaluation exists beyond $m = 5$ to
say how fast that gap closes.

On the audit sequence $\eqref{eq:leading}$ checks exactly: the self-loop part of
$B$ is the same fraction on every graph, the duplicate part equals
$\Phi_5\,S(g)$, and the closed form for $\Phi_m$ agrees with its defining sum.

### The bias on an enumerable case

The audit sequence is $(3, 2, 2, 2, 1)$: heterogeneous, and small enough that all
six of its simple graphs can be enumerated and $P(g)$ evaluated in exact rational
arithmetic, conditioned on completion.

| Quantity | Value |
|:--|:--|
| Max per-graph deviation from uniform | ±2.95% (exactly ±13091/443581) |
| Total variation distance from uniform | 0.0148 |
| Restart mass, retrying sampler | 0.308 |
| Restart rate, repeated configuration model | 0.695 (exactly 73/105) |
| Monte Carlo of the implementation, 200,000 builds | max abs. $z$ vs exact = 1.59 |

The deviation is structured, not diffuse: the three under-sampled graphs are
exactly those containing the edge between the degree-3 and the degree-1 node.
That is what $\eqref{eq:leading}$ predicts. The three degree-2 nodes are
interchangeable, so the six graphs form two orbits of three, with $S = 7$ and
$S = 6$; the hub–leaf edge contributes nothing to $S$ because the leaf has
excess degree zero. With $\Phi_5 = 13/350$ the predicted deviation is
$\pm\Phi_5/2 = \pm 1.86\%$ against the exact $\pm 2.95\%$; the gap is
higher-order terms. The symmetric ± split is a property of the sequence, not of
the sampler: two equal orbits summing to one must sit at equal distances either
side of uniform.

The Monte Carlo run validates the one step the reduction does not give for free —
that after a commit without a reshuffle, the remaining stub order is still
uniform.

The repeated configuration model's restart rate comes from the same reference as
the closed-form check. Summing $16/315$ over the six graphs gives $32/105$, the
probability that uniform pairing is simple, so the repeated model restarts with
probability $73/105$, the table's 0.695.

The two restart rates are rejection rates of different events — the costs being
traded, not two measurements of one quantity. The repeated model discards a draw
whenever the pairing is not simple; retrying discards only builds that reach a
non-graphical residual.

One limit on what this establishes. The exact bias is evaluated only at
enumerable size, in a regime where collisions are frequent. At larger sizes only
the leading-order term is measured, against uniform samples rather than against
the sampler itself; no exact evaluation exists at production sizes.

## Evidence

Degree exactness: measured across all seven test families, 100% of nodes exact on
every build. [`control_audit.py`](evidence/control_audit.py) →
[`control_audit_results.json`](evidence/control_audit_results.json).

Sampler uniformity: the exact evaluation of the chain, the closed form, the
leading-order split and its closed form, the size scan against uniform samples,
and the implementation Monte Carlo.
[`uniformity_bias.py`](evidence/uniformity_bias.py) →
[`uniformity_bias_results.json`](evidence/uniformity_bias_results.json).

!!! info "Reference"
    van der Hofstad, R. *Random Graphs and Complex Networks*, Vol. 1, Chapter 7,
    Configuration Model — Definition 7.5, Proposition 7.7, §7.4 and equation
    (7.5.2). This page derives only what the bias argument needs; the chapter
    gives the full treatment of the configuration model, including the
    probability that a uniform pairing is simple and the repeated model built
    on it. Lecture-notes edition at
    [rhofstad.win.tue.nl/Cap_Sel_Connectivity_in_RG.html](https://rhofstad.win.tue.nl/Cap_Sel_Connectivity_in_RG.html).

    On the sampler's asymptotics: Steger, A. and Wormald, N. C., *Generating
    random regular graphs quickly*, Combinatorics, Probability and Computing 8
    (1999) 377–396. Kim, J. H. and Vu, V. H., *Generating random regular
    graphs*, STOC 2003. Bayati, M., Kim, J. H. and Saberi, A., *A sequential
    algorithm for generating random graphs*, Algorithmica 58 (2010) 860–910.

## Where in the code

- `ConfigModelGraph.from_config` — the builder; the degree assertion runs on every
  call.
- [Configuration model API](../api/configuration-model.md) ·
  [Sequential conditional sampling](../concepts/sequential-conditional-sampling.md)
