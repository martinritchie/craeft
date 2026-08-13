$$T_1 = (G_1(x), S1),$$
$$T_2 = (G_2(x), S1),$$
$$T_3 = (G_1(S1) + G_1(S1) + 2G_1(S1)),$$
$$T_4 = (G_1(S1) + G_2(S1)).$$

To generate the above identities, we consider a susceptible node in position $s$, and list all possible identities of its neighbors. We consider a random susceptible node with in-degree $d_n$ to have been exposed to infection. $\hat{T} = (\hat{T}_1, \hat{T}_2, \hat{T}_3, \hat{T}_4)$ can now be used to determine the probability that a susceptible node has an infectious neighbour within a certain subgraph type. This is done by dividing $\hat{T}^{(s)}$ by the number of states that involve a susceptible at position $s$.

$$\cdot \frac{1}{\sum_{i \in \{A, B, C, D\}} G_{s,i}(s_i, \ldots, \tilde{s}_n, \ldots)}$$

The expected degree of a susceptible node at position $s$, is given by

$$\langle k_s \rangle = \sum_{k=0}^{\infty} k \prod_{i}^{s} \hat{T}_i \bigg|_{s=1}$$

where $\hat{T} = (\hat{T}_1, \hat{T}_2, \hat{T}_3, \hat{T}_4)$. To compute the expected degree for every position of every subgraph, we can take the Jacobian of $\hat{T}$ evaluated at $s = 1$:

$$J = \frac{\partial \hat{T}}{\partial s}\bigg|_{s=1}$$

The $i$th entry of this vector evaluated at $s = 1$. A susceptible node in position $s$ will have remained susceptible from time 0 to time $T_f/J$. This information may be used to form the following equation:

$$\frac{d}{dt}(\hat{J}(t)) = \theta(t) \cdot \hat{J}(t) \bigg|_{s=1} \tag{8}$$

$\theta(t)$ decays at the rate at which a susceptible node its to its node in position $s_1$, conditional on that node being susceptible.

Once a node is newly infected it becomes necessary to determine what, if any, subgraph states are created or destroyed. To do this, we use the susceptible nodes' excess degree prior to the infection. For the full derivation of susceptibles' excess degree refer to Appendix 6.1. In this derivation, the excess degree must be generalised to account for the degree of the different positions occupied may be in, i.e. $d_i$, $i = 1, \ldots, m$. The expected excess degree for susceptible nodes is given by

$$A_{i,j} = k_{ij} \frac{P_j(\tilde{k})}{P_j(\tilde{k})} \bigg|_{\tilde{k}(t)}$$

![Diagram showing network subgraph structures with nodes labeled S and I connected by edges, arranged in multiple configurations corresponding to subgraph types A, B, C, D](figure)
