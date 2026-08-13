There are two key ways in which this work may be extended: (a) generalisation to $5/3$ dynamics. Due to the definition of $P(i)$ it is currently not possible to apply this model to SIS dynamics. However, all the framework relating to network structure is independent from this variable and may therefore remain unchanged. (b) The subgraph approach is highly suitable for adaptation to household models. Household models typically specify a distribution of household sizes overlaid on a contact network to capture the locally dense connectivity present in groups which interact closely [33]. A successful incorporation of such network in our framework could lead to a highly relevant set of household models.

**Acknowledgements** Martin Ritchie acknowledges funding for his PhD studies from EPSRC (Engineering and Physical Sciences Research Council), EP/K503187/1 and the University of Sussex.

**Open Access** This article is distributed under the terms of the Creative Commons Attribution 4.0 International License (http://creativecommons.org/licenses/by/4.0/), which permits unrestricted use, distribution, and reproduction in any medium, provided you give appropriate credit to the original author(s) and the source, provide a link to the Creative Commons license, and indicate if changes were made.

## 6 Appendix

In this Appendix we (a) give a more detailed explanation of the excess degree, (b) provide ODEs for an example network, (c) show how our generalised model reduces to a previous model under specific conditions, (d) provide the derivation of the Jacobian matrix, (e) give pseudocode for both the subgraph-based configuration model and the algorithm used to obtain the data from the networks and finally, (f) compare the SIR dynamics on two configuration model networks with their degree distributions being different but with the same mean and variance.

### 6.1 Excess degree

Recall the probability generating function (PGF) of a network's hypernode degree distribution with $n$ nodal positions:

$$g(x) = \sum_{k} \prod_{j=1}^{n} x_j^{a_j} p_k \tag{11}$$

where $\mathbf{x} = (x_1, x_2, \ldots, x_n)$ is a placeholder, and $\mathbf{k} = (y_1, y_2, \ldots, y_n)$ such that $y_j$ denotes the number of edges incident to nodal position $j$ in the hypernode. The PGF of the excess degree distribution is a critical component in our derivation and it is obtained here as follows: to compute the expected excess degree, select a node at random but proportional to its number of $x_i$ hyperedges, $y_i p_k$. Next, to obtain the expected $x_i$ degree, we must sum over all nodes:
