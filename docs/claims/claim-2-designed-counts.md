# Claim 2 — Subgraph counts are set by the input, not by luck

**The number of triangles, squares or diamonds you design is the number the generator
builds: realized count = designed count + the by-product floor.**

## What is held

| | |
|:--|:--|
| **Exact** | Designed counts, when participation is *prescribed*. Additivity (realized = designed + floor) holds within 2% across every family tested. |
| **Approximate** | Designed counts under *sampled* participation: the degree budget caps each node's participation, and where the cap bites you get fewer shapes than the nominal rate asks for — measured at 62% of nominal in the worst configuration studied (the zero-slack regime). |
| **Not controlled** | The floor itself. It cannot be removed, only predicted and designed above. |

## The floor is predictable

The by-product floor is a closed-form pre-flight calculation, not a build-and-see. The
driver is the mean excess degree
$\kappa = \langle k(k-1)\rangle / \langle k\rangle$, and the expected floor for
length-$L$ cycles is

$$\frac{\kappa^L}{2L}.$$

This one expression reproduces every floor measured in this project to within ~16%. It
over-predicts under heavy-tailed degrees, which is the safe direction: it never
promises a contrast the graph will not deliver. $\kappa$ — not the shape of the degree
distribution — is the controlling variable: two families with the same $\kappa$
produce statistically identical floors at every cycle length.

Cliques are the exception in the best way: the $K_5$ floor is exactly zero in every
regime measured, and $K_4$'s is zero at light-tailed degrees. Under a heavy tail the
$K_4$ floor leaves zero and grows with graph size — measure it at the size you intend
to use.

## Practical consequence

If exact counts matter, **prescribe**; if sampling, **leave headroom** under the
degree budget. At equality the cap clips participation, and prescribing the full
amount instead deadlocks the build — no free stubs remain for pairing.

## Evidence

Additivity and the cap: [`control_audit.py`](evidence/control_audit.py) →
[`control_audit_results.json`](evidence/control_audit_results.json). Floor-formula
validation and the heavy-tail regimes:
[`audit_section_i.py`](evidence/audit_section_i.py) →
[`audit_section_i_results.json`](evidence/audit_section_i_results.json).

## Where in the code

- `SubgraphSequence` — participation input, prescribed (`orbit_counts`) or sampled
  (`distribution`).
- `predicted_cycle_floor`, `mean_excess_degree` — the pre-flight calculation.
- [Metrics API](../api/metrics.md) ·
  [Configuration model API](../api/configuration-model.md)
