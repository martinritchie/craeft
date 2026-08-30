# Claim 1 — The degree sequence is exact

**The number of edges (stubs) originating from each node is exact.**

## What is held

| | |
|:--|:--|
| **Exact** | Realized degree = target degree, per node, every build. Asserted at build time: a violation raises rather than returning a graph. |
| **Approximate** | Uniformity of the sampler over the space of compatible graphs (~1.5% bias per graph; see below). |
| **Not controlled** | Everything else — which is the point of [Claim 4](claim-4-residual-freedom.md). |

## Why it holds

During construction stub pairings that would create a self-loop or a duplicate edge are put back and
redrawn, never deleted. Alternatively, dropping the colliding pair
silently loses edges (roughly 27 per thousand-node build at typical densities) and
poisons every downstream measurement, because the "matched" degree sequences are no
longer matched. For non-graphical stub pools, those where no eligble pairings remain, the process starts afresh. 

## The cost, quantified


Retrying collisions makes the sampler slightly non-uniform over the possible graphs:
graphs reachable by fewer collision paths are mildly favoured. Measured, the bias is
~1.5% per graph and does not shift triangle counts detectably. Both sides of a matched
pair share the same bias, so it cancels in the comparison a benchmark actually makes.

> MR: Its not clear what ~1.5% per graph means here.


## Evidence

Measured across all seven test families: 100% of nodes exact on every build. The runs
live in [`control_audit.py`](evidence/control_audit.py) with exact values in
[`control_audit_results.json`](evidence/control_audit_results.json).

> The claim of exact degree sequences should be supported with an `assert` or a validator. 
> The 1.5% will require empirical evidence or a mathematical result, ideally the former validated against the latter.

## Where in the code

- `ConfigModelGraph.from_config` — the builder; the degree assertion runs on every
  call.
- [Configuration model API](../api/configuration-model.md) ·
  [Sequential conditional sampling](../concepts/sequential-conditional-sampling.md)
