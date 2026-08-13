# Related Papers (2025-2026)

Recent papers where craeft's CMA algorithm could directly contribute, ordered by
estimated impact.

---

## 1. Do Graph Diffusion Models Accurately Capture and Generate Substructure Distributions?

- **Authors:** Xiyuan Wang, Yewei Liu, Lexi Pang, Siwei Chen, Muhan Zhang
- **Date:** February 2025 (under review)
- **Link:** https://arxiv.org/abs/2502.02488

**Problem:** Graph diffusion models (DiGress, GDSS, etc.) fail to preserve the
substructure count distributions of training data when generating new graphs. The root
cause is that Graph Transformer backbones lack the expressivity to model these
distributions. Swapping in more expressive GNN backbones helps but doesn't solve the
fundamental problem.

**CMA contribution:** CMA generates graphs with exact, prescribed motif counts —
providing rigorous ground truth that the field currently lacks. A CMA-generated
benchmark suite would give any graph generation method a clean target to evaluate
against. CMA could also serve as a statistical baseline: "can your diffusion model
match what a principled statistical generator produces?"

**Impact:** Highest. Sits at the intersection of diffusion models and GNN
expressivity — the two hottest graph ML topics. The open wound this paper identifies
(no structural fidelity in generation) is exactly where CMA operates.

---

## 2. Studying and Improving Graph Neural Network-based Motif Estimation

- **Authors:** Pedro C. Vieira, Miguel E. P. Silva, Pedro Manuel Pinto Ribeiro
- **Date:** June 2025
- **Link:** https://arxiv.org/abs/2506.15709

**Problem:** Motif significance profile (SP) estimation via GNNs is under-explored
with no established benchmarks. The authors frame SP estimation as multitarget
regression and find that 1-WL limited models can approximate the generation process of
synthetic graph generators by comparing predicted SPs against known generators. But
there is no benchmark suite of graphs with known motif profiles.

**CMA contribution:** Almost tailor-made. CMA is the synthetic generator with known
motif profiles that this paper needs for validation. A benchmark suite of CMA-generated
graphs with prescribed motif distributions would directly serve as the missing
evaluation infrastructure.

**Impact:** High within the motif estimation community. Narrower audience than paper 1
but the fit is near-perfect.

---

## 3. HOG-Diff: Higher-Order Guided Diffusion for Graph Generation

- **Authors:** (Feb 2025)
- **Link:** https://arxiv.org/abs/2502.04308

**Problem:** A direct response to the substructure distribution failure in diffusion
models. Adds higher-order structural guidance (triangle counts, clustering
coefficients) to the diffusion process. But guidance signals are computed during
generation — there is no principled way to prescribe target higher-order statistics
upfront.

**CMA contribution:** CMA provides the missing piece — a way to specify target motif
distributions and generate graphs that satisfy them. HOG-Diff's guidance could be
conditioned on CMA-derived targets rather than heuristic signals, creating a hybrid
statistical-learned generation pipeline.

**Impact:** Medium-high. Directly builds on paper 1's findings. A CMA + diffusion
hybrid would be novel.

---

## 4. Recent Developments in GNNs for Drug Discovery (Survey)

- **Date:** June 2025
- **Link:** https://arxiv.org/abs/2506.01302

**Problem:** GNN-based molecular generation is categorised into unconstrained,
constrained (targeted substructures), and ligand-protein-based. Constrained generation
with targeted substructures remains difficult — models struggle to guarantee specific
functional group presence.

**CMA contribution:** Functional groups are motifs. CMA's ability to prescribe motif
distributions maps to "generate a molecular graph containing exactly N aromatic rings
and M hydroxyl groups." Not as a molecular generator itself, but as the structural
constraint engine that learned models are conditioned on.

**Impact:** Highest commercial relevance (drug discovery). Requires the most additional
work — needs an RDKit molecular graph integration layer before CMA can operate on
molecular representations.

---

## 5. The Simpliciality of Higher-Order Networks

- **Authors:** (EPJ Data Science, 2024, corrected 2025)
- **Link:** https://epjdatascience.springeropen.com/articles/10.1140/epjds/s13688-024-00458-1

**Problem:** Existing generative models for higher-order networks fail to capture the
level of simpliciality displayed by empirical data. Simpliciality measures whether
higher-order interactions satisfy downward closure — if a triangle exists, all its
edges must exist too. Current models don't get this right.

**CMA contribution:** CMA already enforces structural constraints on motif placement.
Extending it to enforce simplicial closure constraints (if a K4 is placed, all its
constituent triangles must also be present) would directly address this open problem.
The decompose-match-connect architecture is already set up for hierarchical motif
constraints.

**Impact:** Niche but growing. Higher-order networks are an active research area and
this is a concrete, well-defined limitation that CMA's architecture is suited to
address.

---

## Summary

| # | Paper | Year | CMA Fit | Additional Work Needed |
|---|-------|------|---------|----------------------|
| 1 | Diffusion substructure fidelity | 2025 | Direct | PyG export + benchmark script |
| 2 | GNN motif estimation | 2025 | Direct | Benchmark dataset generation |
| 3 | HOG-Diff guidance | 2025 | Natural | Conditioning interface |
| 4 | Drug discovery GNNs | 2025 | Bridge | RDKit integration |
| 5 | Higher-order simpliciality | 2024/25 | Extension | Simplicial closure in CMA |
