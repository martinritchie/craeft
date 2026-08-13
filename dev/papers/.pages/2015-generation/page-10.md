multi-edges, these results also imply that the number of self and multi-edges reduce as the number of $G_c$ increases.

We believe the following to be an intuition behind this surprising result: consider a stub incident to two $G_c$ stubs. For a self-loop to be made by this node, there is a single opportunity: its hyperedge must be simultaneously selected during the connection procedure. Now, if we consider a node with the same degree, but with each of its stubs being used to form only lines, then there are $k(k-1)/2 \approx k^2$ different ways in which pairs of stubs may be selected that result in a self-edge. Thus, in general, hyperedges will reduce the number of ways in which tuples of stubs result in connected, connected-to stubs, and this will impact both the self and multi-edge probabilities.

**Edge probability:** With the subgraph connection process it is possible to replicate some of the estimates that have been previously produced for the standard configuration model and to extend them significantly. The following calculations are intended to further develop intuition and by no means form a rigorous argument. Let us consider a network model composed of only $G_c$ subgraphs, referred to henceforth as the $G_c$ model. Let $5m_c$ denote the total number of $G_c$ hyperedges, i.e., this network has a total of $m_c$ $G_c$ subgraphs and $5m = 2m_c$ stubs, since each $G_c$ hyperedge is composed of two stubs.

We first consider the probability of two nodes sharing a single edge in the $G_c$ model. Let nodes $i$ have $G_c$ degrees of $k_i$ and $k_j$, respectively. A single hyperedge of $i$ may connect to any of the $k_j$ hyperedges originating from $j$. The probability of selecting one of $j$'s hyperedges is $k_j/(5m_c - 1)$, since we can no longer select the initial hyperedge incident to $i$. The remaining $k_j - 1$ hyperedges of $j$ that were not selected are then connected to one of the remaining $5m_c - 3$ stubs. The remaining stub of $i$, once selected, incidents to a third distinct node, i.e., any one of the $5m_c - 1 - k_j$ hyperedges that are not incident to either $i$ or $j$. If a hyperedge incident to $i$ and $j$ also results in both a self and multi-edge. Therefore, the probability of $i$ and $j$ sharing a single edge is given by

$$p_{ij} \sim \frac{k_i k_j}{5m_c - 1} \left(1 - \frac{k_j - 1}{5m_c - 3}\right) \tag{4}$$

Since the degree distribution is fixed and we are interested in the limit as $N \to \infty, m_c \to \infty, m \to \infty$,

$$\lim_{5m_c \to \infty} p_{ij} = \frac{k_i k_j}{5m_c} \tag{5}$$

Let us now consider that in the $G_c$ model each node is incident to $2G_c = k_i$ stubs in a network composed of a total of $2(5m_c) = 2m$ stubs. Therefore, the subgraph configuration equation (5) the edge probability of the $G_c$ model can be compared to its equivalent configuration model

$$\frac{k_i k_j}{5m_c} = \frac{k_i k_j}{2m}$$

where the r.h.s. represents the edge probability in the configuration model. This counter-intuitive result for the $G_c$ model is due to half of a node's stubs being obliged to connect to a third distinct node, including probabilities of $i$ and $j$ connecting which would otherwise be possible in the configuration model.

**Multi-edge probability:** As in the standard configuration model, we can use equation (5) to estimate the probability of multi-edges that may happen in two ways: being either (a) a triple of nodes, with at least one of the constituent pairs already being connected or (b) all three of nodes already being connected. We first consider (a), the more likely scenario: $i$ and $j$ will share an edge with the probability given as in
