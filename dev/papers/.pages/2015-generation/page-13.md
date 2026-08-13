which, after summing over all nodes, yields

$$\sum_{i} \frac{s_i(t-1)}{b_m} \cdot \frac{\theta(G_i - \tilde{G}_i)}{\tilde{Z}(G_i)} \tag{11}$$

Since the determining factor of this expectation is the selection of a pair of hyperedges incident to the same node, we shall compare it to the edge probability in the equivalent configuration model network:

$$\frac{\theta(G_i - \tilde{G}_i)}{\tilde{Z}(G_i)} \approx \frac{\tilde{s}_i(t) \cdot (2\tilde{Z}(G_i) - \tilde{s}_i(t))}{\tilde{Z}(G_i)} \tag{12}$$

i.e., as $N \to \infty$ we expect that the number of duplicate node selections resulting from $G_c$ placement will be strictly less than the number in the equivalent configuration model network.

It is worth noting that this algorithm does not prevent the creation of subgraphs with overlap, making it possible to create subgraphs with overlap. In this case if the overlap on a particular subgraph was collapsed down to a single edge, the process would yield a $G_0$ subgraph. The expected number of these events was shown to be bounded by a number of multi-incident configuration-model subgraphs in the network. However, this type of connection was not permitted in our implementation.

Additionally, we can only provide estimates regarding the frequency of "erroneous" subgraphs, that is, $G_c$ subgraphs that appear beyond that which were controlled for. This type of connection is permitted in our implementation and would result in the subgraph by-products as shown in Figure 3. However, Fig. 2 indicates that the number of erroneous $G_c$ subgraphs decreases as the number of intended subgraphs increases.

## 2.4 The Ry algorithm

The Big-V algorithm does not generate networks as such, but is a widely-used, see [12, 27–29] for example, degree-preserving rewiring algorithm, making it possible to control clustering. At each iteration, the algorithm selects a linear chain of five nodes at random, e.g. $(a, b, c, d, e)$ with four edges $(a, b), (b, c), (c, d), (d, e)$. It then deletes edge $(b, d)$ and reforms it to form $(a, e)$ to form $(a, b), (b, c), (c, d), (d, e)$ and $(a, e)$. When starting from a rewired network, this augmentation increases the local clustering coefficient. This process is repeated until the desired level of clustering is achieved. It is possible to include a Metropolis-style augmentation whereby at each step the local clustering coefficient is computed for the five nodes before and after rewiring, and the rewired configuration is only accepted if it results in an increase in average local clustering. It is worth noting that this algorithm leads to a positive degree-degree correlation that was not necessarily present in the original network.

In this article, we use the Big-V algorithm to demonstrate that our newly proposed algorithms are able to sample from a larger part of the state space of all possible networks with a given degree sequence and global clustering coefficient.
