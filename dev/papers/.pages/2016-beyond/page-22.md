$R = \gamma \hat{t},$

where $\hat{t}$ is the probability generating function that generates the hypernode degree distribution and $\hat{\beta} = (\hat{\gamma}_1, \hat{\gamma}_2, \hat{\gamma}_3, \hat{\gamma}_4, \hat{\gamma}_5)$ is the probability that infection via subgraphs of types one to five has not been transmitted. The total system size for this example network is given by

$$3^2 + 3^1 + 5 \times 2 = 43,$$

with each term in the above corresponding to $G_0$, $G_2$, survivor functions and epidemic prevalence, respectively. In general, the total number of equations is given by:

$$\sum_{i=1}^{m} |G(i)| = |G_{\xi}| + 2,$$

where $G_{\xi}$ denotes a subgraph, $|G_i|$ is the number of nodes in a subgraph, and $m$ is the total number of subgraphs.

## 6.3 Equivalence to previous model for complete subgraphs

The PGF formulation originally proposed by Volz et al. (2011) is equivalent to our proposed model in the case of complete subgraphs. Consider an arbitrary complete subgraph composed of nodes and a network that is composed only of this subgraph. If positions within the subgraph are labelled explicitly, $\{v_1, \ldots, v_s\}$, as we have done in our approach, then the PGF of such a network is given by

$$\hat{p}_k(k) = \sum_k p_k k^s, \tag{14}$$

where $\hat{s} = (y_1, \ldots, y_{s-1})$. Volz et al.'s framework treats all topologically equivalent positions as one single position. Thus, in this case, the subgraph has a single label, $y$, that corresponds to a single count, $y$, and the PGF takes the following form:

$$\hat{p}_y(y) = \sum_k p_k y^k. \tag{15}$$

We now show how one may obtain Eq. (15) from Eq. (14). Since both PGFs describe the same type of network, in which our formulation allocates position $y_i$ must be $1/k$ the rate at which Volz et al.'s formulation allocates $y$. If we replace the unique position labels of Eq. (14) with a single position marker (such as in Volz et al.'s model), the following expression is obtained:
