# Claim 1 — The degree sequence is exact

**Every node ends with exactly the degree you asked for. Not on average. Not
approximately. Element-wise, on every build.**

## What is held

| | |
|:--|:--|
| **Exact** | Realized degree = target degree, per node, every build. Asserted at build time: a violation raises rather than returning a graph. |
| **Approximate** | Uniformity of the sampler over the space of compatible graphs (~1.5% bias per graph; see below). |
| **Not controlled** | Everything else — which is the point of [Claim 4](claim-4-residual-freedom.md). |

## Why it holds

Stub pairings that would create a self-loop or a duplicate edge are put back and
redrawn, never deleted. The tempting alternative — dropping the colliding pair —
silently loses edges (roughly 27 per thousand-node build at typical densities) and
poisons every downstream measurement, because the "matched" degree sequences are no
longer matched. If the stub pool deadlocks, the builder resets and rewires from
scratch instead of compromising.

## The cost, quantified

Retrying collisions makes the sampler slightly non-uniform over the possible graphs:
graphs reachable by fewer collision paths are mildly favoured. Measured, the bias is
~1.5% per graph and does not shift triangle counts detectably. Both sides of a matched
pair share the same bias, so it cancels in the comparison a benchmark actually makes.

## Evidence

Measured across all seven test families: 100% of nodes exact on every build. The runs
live in [`control_audit.py`](evidence/control_audit.py) with exact values in
[`control_audit_results.json`](evidence/control_audit_results.json).

## Where in the code

- `ConfigModelGraph.from_config` — the builder; the degree assertion runs on every
  call.
- [Configuration model API](../api/configuration-model.md) ·
  [Sequential conditional sampling](../concepts/sequential-conditional-sampling.md)
