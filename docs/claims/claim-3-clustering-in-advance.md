# Claim 3 — Clustering is known before the graph exists

**Designed clustering is a closed-form calculation from the inputs. No build
required.**

Clustering here means the global coefficient, or transitivity,

$$C \;=\; \frac{3N_\triangle}{N_{\text{triples}}}, \qquad
N_{\text{triples}} \;=\; \sum_i \binom{k_i}{2},$$

not the average of per-node local coefficients. The two differ on any graph
whose triangles sit unevenly across degrees, and only the global one is
pinned by what follows.

## Why it holds

Because the degree sequence is pinned ([Claim 1](claim-1-exact-degrees.md)),
the denominator is fixed in advance. Clustering therefore reduces to a
triangle count. With orbit resolution every shape's triangle contribution is
a known integer: a triangle brings one, a diamond corner one or two depending
on the orbit, an empty cycle exactly zero. Add up the designed shapes and you
have the designed clustering. `designed_clustering(config)` computes it.

## What is held

| | |
|:--|:--|
| **Exact** | The denominator. The designed triangle count, when participation is *prescribed*. Under *sampled* participation the designed count is an expectation taken at the nominal rate, and it is only right when the degree cap does not bind (see [Claim 2](claim-2-designed-counts.md)); where the cap clips, the realized count falls below it by the clipped fraction. |
| **Approximate** | The realized value. Realized triangles equal designed plus the by-product floor to within 1% at regular degrees ($n = 1200$, degree 12, 8 builds; ratios 0.998, 1.010, 0.996, 0.994 for the triangle, diamond, $K_4$ and square families). Realized clustering is designed clustering plus the floor's share of the denominator, and the floor is not small: in that audit it is 220 triangles against designed counts of 0 to 600. |
| **Not controlled** | Local clustering per node. Prescribing the orbit split pins each node's *designed* triangles exactly, but its realized value adds a local floor that grows with degree ([Claim 4](claim-4-residual-freedom.md)). |

The audit's designed and realized values, side by side:

| Family | Designed triangles | Floor | Realized | $C$ designed | $C$ realized |
|:--|--:|--:|--:|--:|--:|
| 200 triangles | 200 | 220.5 | 419.8 | 0.0076 | 0.0159 |
| 150 diamonds | 300 | 220.5 | 525.5 | 0.0114 | 0.0199 |
| 150 $K_4$ | 600 | 220.5 | 817.5 | 0.0227 | 0.0310 |
| 150 squares | 0 | 220.5 | 219.3 | 0 | 0.0083 |

So a matched pair is matched on `designed_triangles` (or on designed plus
predicted floor), and both members share the same floor because they share
the same degree sequence. Comparing one family's realized $C$ with another's
designed $C$ compares different things.

## Two consequences worth internalising

**Empty cycles are triangle-free by construction.** A family built from
squares, pentagons or hexagons matches the random graph's clustering for
free: the square family above realizes exactly the floor.

**The triangle is the only cycle that *is* clustering.** On the
heterogeneous cycle families ($n = 1000$, 12 builds) the triangle family
realizes $C = 0.131$ against 0.0071 to 0.0074 for the square, pentagon and
hexagon families and 0.0058 for the null: about 18 times any other cycle.
Use triangles only when clustering is the variable under study.

## Evidence

Closed-form versus realized, the floor and the per-family table:
[`control_audit.py`](evidence/control_audit.py) →
[`control_audit_results.json`](evidence/control_audit_results.json),
section `orbit_clustering` (8 builds) and section `cmodels` (12 builds).

!!! info "Reference"
    On the two clustering coefficients and why they differ: Newman, M. E. J.,
    *The structure and function of complex networks*, SIAM Review 45 (2003)
    167–256, §3.2.

## Where in the code

- `designed_clustering`, `designed_triangles` — the closed-form calculation.
- `global_clustering_coefficient` — the realized value; `local_clustering`
  gives the per-node coefficients.
- `SubgraphSequence.orbit_counts` — prescribing the orbit split pins per-node
  designed triangles exactly.
- [Metrics API](../api/metrics.md) ·
  [Clustered Configuration Model](../concepts/ccm.md)
