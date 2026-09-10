# Claim 2 — Subgraph counts are designed, plus a predictable floor

**The number of designed subgraphs is set by the input. The realized count is
the designed count plus a by-product floor, and the floor can be predicted
before the build.**

## What is held

| | |
|:--|:--|
| **Exact** | The designed count, when participation is *prescribed*: `orbit_counts` pins the number of instances of each shape. |
| **Approximate** | Additivity, realized = designed + floor: within 2% on the five regular-degree families audited (triangle, square, pentagon, hexagon, $K_4$ at $n = 1000$, degree 10 or 12, 12 builds each; ratios 0.996, 1.018, 0.999, 1.001, 0.985). Additivity has not been measured for heterogeneous degrees beyond the triangle count. Designed counts under *sampled* participation: the degree budget caps each node's draw, so where the cap binds fewer shapes are placed than the nominal rate asks for. On the regular-degree-4, rate-2 family the audit records 487 designed instances against a nominal 667, i.e. 73%. |
| **Not controlled** | The floor. It cannot be removed, only predicted and designed above. |

## The floor is predictable

Shapes also form by accident when stubs are paired at random. For cycles the
expected number of accidental $L$-cycles in a configuration model with mean
excess degree

$$\kappa \;=\; \frac{\langle k(k-1)\rangle}{\langle k\rangle}$$

is, to leading order in $n$,

$$\mathbb{E}[N_L] \;\approx\; \frac{\kappa^L}{2L}.$$

$\kappa$ is the number of onward edges found after arriving at a node along a
random edge. The formula is classical: Bollobás (1980) and Wormald (1981) for
regular graphs, where $\kappa = d - 1$, and Bianconi and Marsili (2005) for a
general degree sequence. `predicted_cycle_floor` evaluates it from the degree
sequence alone, so whether a design is measurable is a pre-flight
calculation, not a build-and-see.

**Where it comes from.** In the configuration model two nodes $u, v$ are
joined with probability close to $k_u k_v / 2m$, and a node entered along one
edge has $k_u - 1$ further stubs. A particular cycle through $v_1, \ldots,
v_L$ therefore appears with probability close to
$\prod_i k_{v_i}(k_{v_i}-1) / (2m)^L$. Summing over ordered choices of $L$
distinct nodes gives $\big(\sum_v k_v(k_v-1)\big)^L / (2m)^L = \kappa^L$,
and dividing by $2L$ for the $L$ starting points and two directions gives the
result.

**Two things to know before using it.**

- It counts *all* cycles. The audits count *induced* (chordless) cycles,
  which are fewer, and the gap grows with density.
- It is asymptotic in $n$. Bianconi and Marsili give the range of validity
  for power-law degrees with exponent $2 < \gamma < 3$ as $L \ll
  n^{(3-\gamma)/2}$, which is about 5.6 at $\gamma = 2.5$, $n = 1000$.

**How well it holds.** Predicted over measured floor, from the shipped audit:

| Degrees | $\kappa$ | $n$, builds | $L=3$ | $L=4$ | $L=5$ | $L=6$ |
|:--|--:|:--|--:|--:|--:|--:|
| regular 4 | 3.0 | 1000, 16 | 1.01 | 1.07 | 0.97 | 1.04 |
| regular 6 | 5.0 | 1000, 12 | 1.14 | 0.97 | 1.01 | 1.04 |
| regular 10 | 9.0 | 1000, 6 | 0.99 | 1.00 | 1.04 | 1.06 |
| $2 \times \mathrm{Pois}(2)$ | 4.96 | 1000, 11 | 0.98 | 1.02 | 1.04 | 1.10 |
| power law $\gamma = 3.5$ | 4.94 | 1000, 5 | 1.06 | 1.15 | 1.16 | 1.35 |
| power law $\gamma = 3.5$ | 5.30 | 4000, 5 | 0.97 | 1.00 | 1.03 | 1.09 |
| power law $\gamma = 2.5$ | 7.84 | 1000, 5 | 1.10 | 1.34 | 1.86 | 2.57 |
| power law $\gamma = 2.5$ | 11.77 | 4000, 5 | 1.03 | 1.31 | 1.71 | 2.42 |

At light tails the formula is within about 15% of the measured floor. Under
the heavy tail $\gamma = 2.5$ it over-predicts, by a factor that grows with
$L$ to 2.6 at $L = 6$: that is where the validity bound above is crossed and
where the induced-versus-all gap is largest. Over-prediction is the safe
direction for a design, since the formula never promises a contrast the graph
will not deliver, but it means the heavy-tail floors must be measured, not
predicted.

$\kappa$ is the controlling variable at light tails. The $2 \times
\mathrm{Pois}(2)$ family and the $\gamma = 3.5$ family have nearly the same
$\kappa$ and their measured floors agree within replicate spread at every
length (20.8, 74, 288, 1121 against 19.8, 70, 290, 1108, from 11 and 5
builds; no formal test was run). Under a heavy tail $\kappa$ itself grows
with $n$ (7.8 at $n = 1000$ to 11.8 at $n = 4000$ for $\gamma = 2.5$ with
the $\sqrt{n}$ cutoff), so scaling up raises the floor.

Cliques are the exception. The $K_5$ floor is zero in every regime measured.
The $K_4$ floor is zero at light tails; under $\gamma = 2.5$ it is 0.8 at
$n = 1000$ and 4.6 at $n = 4000$ (5 builds each, per-build counts 1, 1, 1,
1, 0 and 1, 4, 5, 0, 13), so measure it at the size you intend to use.

## Practical consequence

If exact counts matter, **prescribe**. If sampling, **leave headroom** under
the degree budget: at equality the cap clips participation. A prescription
that exceeds any node's budget is rejected before the build by the pre-flight
check (`AllocationError`), not discovered during it.

## Evidence

Additivity and degree preservation: [`control_audit.py`](evidence/control_audit.py)
→ [`control_audit_results.json`](evidence/control_audit_results.json),
sections `subgraph_fidelity` and `degree_preservation` ($n = 1000$, 12
builds per family). Clipping at the regular-degree-4 family:
[`audit_section_h.py`](evidence/audit_section_h.py) →
[`audit_section_h_results.json`](evidence/audit_section_h_results.json),
`h1_local_snr` (12 builds). Floor-formula validation, the heavy-tail
regimes and the clique floors:
[`audit_section_i.py`](evidence/audit_section_i.py) →
[`audit_section_i_results.json`](evidence/audit_section_i_results.json),
sections `i1d_fresh_spotcheck` and `i2_heavy_tails` (build counts as in the
table above).

!!! info "Reference"
    The cycle-count law for random regular graphs: Bollobás, B., *A
    probabilistic proof of an asymptotic formula for the number of labelled
    regular graphs*, European J. Combin. 1 (1980) 311–316; Wormald, N. C.,
    *The asymptotic distribution of short cycles in random regular graphs*,
    J. Combin. Theory B 31 (1981) 168–182. For a general degree sequence:
    Bianconi, G. and Marsili, M., *Loops of any size and Hamilton cycles in
    random scale-free networks*, J. Stat. Mech. (2005) P06005, eq. (30) and
    the validity conditions that follow it.

    The subgraph ensemble itself: Newman, M. E. J., *Random graphs with
    clustering*, Phys. Rev. Lett. 103 (2009) 058701; Miller, J. C.,
    *Percolation and epidemics in random clustered networks*, Phys. Rev. E
    80 (2009) 020901; Karrer, B. and Newman, M. E. J., *Random graphs
    containing arbitrary distributions of subgraphs*, Phys. Rev. E 82 (2010)
    066118, whose subgraph "roles" are the orbits used here.

## Where in the code

- `SubgraphSequence` — participation input, prescribed (`orbit_counts`) or
  sampled (`distribution`).
- `predicted_cycle_floor`, `mean_excess_degree` — the pre-flight calculation.
- [Metrics API](../api/metrics.md) ·
  [Configuration model API](../api/configuration-model.md)
