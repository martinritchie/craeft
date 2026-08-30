# Claim 3 — Clustering is known before the graph exists

**Designed clustering is a closed-form calculation from the inputs. No build
required.**

> MR: Is this exact or approximate? What claims can we make about _local_ clustering?

## Why it holds

Because the degree sequence is pinned ([Claim 1](claim-1-exact-degrees.md)), the
denominator of the global clustering coefficient is fixed in advance:

$$C=\frac{3N_\triangle}{N_{\text{triples}}},\qquad
N_{\text{triples}}=\sum_i\binom{k_i}{2}\ \text{ depends only on the degrees.}$$

So clustering reduces to a triangle count. And with orbit resolution, every shape's
triangle contribution is a known integer — a triangle brings one, a diamond corner
brings one or two depending on the orbit, an empty cycle brings exactly zero. Add up
the designed shapes and you have the designed clustering, in closed form.
`designed_clustering(config)` computes it.

## What is held

| | |
|:--|:--|
| **Exact** | The designed triangle count and the clustering denominator. |
| **Approximate** | The realized value: reality adds the by-product floor on top. Designed + floor predicts realized triangles within 1% at regular degrees; realized clustering lands within ~2% of the design across families. |
| **Not controlled** | *Local* clustering per node — unless the orbit split is prescribed, in which case per-node designed triangle counts are exact too. |

## Two consequences worth internalising

**Empty cycles are triangle-free by construction.** A family built from squares,
pentagons and hexagons matches the random graph's clustering for free — no tuning.

**The triangle is the only cycle that *is* clustering.** Any family including it
varies clustering whether you wanted to or not; it is a ~19× clustering outlier
against every other cycle. Use triangles only when clustering is the variable under
study.

## Evidence

Closed-form vs realized comparisons across all families:
[`control_audit.py`](evidence/control_audit.py) →
[`control_audit_results.json`](evidence/control_audit_results.json).

## Where in the code

- `designed_clustering` — the closed-form calculation.
- `SubgraphSequence.orbit_counts` — prescribing the orbit split pins per-node designed
  triangles exactly.
- [Metrics API](../api/metrics.md) ·
  [Clustered Configuration Model](../concepts/ccm.md)
