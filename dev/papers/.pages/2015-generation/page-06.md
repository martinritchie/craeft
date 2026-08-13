6 M. RITCHIE ET AL.

2.1.1 *A priori clustering calculation* The global clustering coefficient is defined as the ratio between the total number of triangles and the total number of connected triples of nodes, $\Delta = \frac{\text{triangles}}{\text{triples}}$; each triangle contains three triples of nodes: $C = \frac{\Delta_t}{\Delta_t}$. It should be noted that each unique triangle is counted six times and each unique triple is counted three times. The number of triples incident to a node of degree $k$ is given by $\binom{k}{2} = k(k-1)/2$ since a node will form a triple with every pair of its neighbours and each triple is counted twice. The expected number of triples for a node of degree $k$ is therefore obtained by summing $P(K = k) \cdot k(k-1)$ over all degrees, where $P(K = k)$ is the probability of finding a node of degree $k$. The expected number of triangles incident to a node of degree $(s_i, d_i)_i$ can be calculated from the Diophantine equation's solution space associated with that degree. To do this, one needs to sum all occurrences of triangle corners, regardless of what the other nodes belong to. Each unique configuration of $(g_1, g_2)$ is counted once for a single node and needs to be multiplied by the frequency of that configuration at random. Finally we are in a position to compute the expected global clustering coefficient as

$$C = \frac{\langle\Delta_t\rangle}{\langle\Delta_t\rangle} = \frac{\sum_k \frac{P(K=k) \cdot (\text{triangles})}{...}}{\sum_k P(K=k) \cdot k(k-1)}$$

(2)

For example, let us consider the homogeneous network of $N = 5$ and the input subgraphs $G_1$ and $G_2$. These subgraphs induce the vector of coefficients $\mathbf{m} = (1, 2, 3)$ and for $k = 5$, has the following solution space

| $G_1$ | 5 | 3 | 2 | 1 | 0 |
|--------|---|---|---|---|---|
| $g_1$  | 0 | 1 | 0 | 2 | 1 |
| $g_2$  | 0 | 0 | 1 | 0 | 2 |

where the rows give the number of each hyperedge, the columns give an individual solution and $g_1$ and $g_2$ denote the double and triple hyperedge of $G_2$, respectively. From this we may calculate the expected number of triangles: $\langle \Delta \rangle$. In this case we can see, on average for every $g_1$ corner, a DNA will have approximately 2.5 $g_1$ edges per node, and on average $g_2 = 0.4$ per node, meaning that $g_1 = 2.5$ and $g_2 = 0.4$ will be generated in equal quantities. The expected number of $g_2$ is given by the expected number of $g_2 = 2/5$ per node. Since a triangle has a unique corner, the number of $g_1$ corners also gives the total number of triangles, that is uniquely counted and per node. So the expected number of triangle per node is $12/5$, each triple being counted over all triples, and from this we have a theoretical global clustering of $C = 0.12$. Computationally, we verify this by generating such networks with $N = 5000$, and find that the number of open triples and triangles is exactly $\langle \cdot \rangle = 100000$ and $\langle \Delta \rangle = 12120$, resulting in a global clustering of $0.212$, as expected.

## 2.2 *Cardinality matching*

The cardinality matching algorithm (CMA) requires as input a degree sequence, a set of subgraphs and corresponding subgraph sequences, i.e., multiplicity sequences specifying to which and how many subgraph each node belongs to. Note that these sequences are not yet allocated. The CMA algorithm proceeds to allocate hyperedges of subgraphs to nodes that have a sufficient number of stubs to accommodate the subgraph's stubs. The algorithm outputs hyperedge node memberships from which the input degree sequence may be recovered exactly. This then can be used to realise a network based on a modification of the configuration model.
