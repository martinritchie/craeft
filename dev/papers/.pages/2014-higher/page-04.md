number of unique triangles in the network can be determined by $\Psi$

$$[\triangle] = N\left(\frac{\Psi_{3}(1,1,1,1)}{4} \cdot \frac{4 \cdot \Psi_{4}(1,1,1,1)}{\Psi_{3}(1,1,1)}\right), \tag{4}$$

since each square is quadruply counted and contains four separate triangles. Clustering is measured as the ratio of three times the number of triangles to all closed and unclosed triples:

$$\phi_{\text{global}} = \frac{3N\left(\frac{\Psi_{3}(1,1,1)}{2}\right) - \Psi_{3}(1,1,1)}{N\varphi(1)}, \tag{5}$$

$$\phi_{\text{local}} = \frac{\Psi_{4}(1,1,1,1)}{N\Psi_{3}(1,1,1,1)/N} \cdot \frac{1}{\varphi(1)}, \tag{6}$$

$$\phi_{\vec{p}} = \frac{p_0 + 2p_1}{3}. \tag{7}$$

For the two types of CCM networks used in this study: $p_1 = 0.5$, $p_2 = 0.5$ yield $\phi = 0.2$ and $p_1 = 1$ yields $\phi = 0.4$ (see Table 2).

## 2.2. *Network metrics: third and higher-order network structure*

Here we give a succinct summary of the classic and newly proposed network metrics that will be used to compare and contrast the networks resulting from the different algorithms. Although the novelty of the paper is around order-four structure, we will first consider classic (or third-order) network measures, such as clustering in the global sense as well as distribution of clustering at the node level, nodal betweenness centrality and connected component analysis via percolation. We then augment the classic description of networks with an analysis of the distribution of motifs of order higher than closed and open triples both globally and on a per node basis. A network of $N$ individuals is represented with an adjacency matrix, $A_{ij} \in \{0, 1\}^N$. A pair of individuals $(i, j)$ share a connection if $A_{ij} = 1$. The networks are undirected, $A = A^T$, and self-loops are not allowed $A_{ii} = 0, \forall i, N$.

1. *Clustering: clustering may be defined in two ways (Watts and Strogatz, 1998): local (node level) and global (network level). The local clustering of a node $n$, of degree $m_n$, is the ratio of connections between neighbours of $n$ and potential connections of neighbours of $n$. Let $A^2$ denote the sub-adjacency matrix of the neighbourhood of $n$ then*

$$\phi_{\text{local}}^n = \frac{\sum_{ij} V_{ij}^n}{m_n(m_n - 1)} \tag{8}$$

Global clustering is defined as the ratio of the total number of closed triples to the total number of connected structures with 3 nodes. This may be computed from the adjacency matrix as (Keeling, 1999b):

$$\phi_{\text{global}} = \frac{6 \cdot \text{trace}(A^3)}{||A^2|| - \text{trace}(A^2)}, \tag{9}$$

where $||A^2||$ denotes the sum of all elements of $A^2$. Manipulating the adjacency matrix in this way yields multiplicative counts. An alternative method to obtain the equivalent counts is as follows:

$$[v] = \frac{1}{2} \sum_{i} m_i(m_i - 1), \tag{10}$$

yielding all connected structures of 3 nodes (closed and unclosed), where

$$[\triangle] = \sum_{i < j < k} A_{ij} A_{jk} A_{ki}, \tag{11}$$

yielding six times the number of unique triangles. A more complete description of this approach is provided in *Appendix A.3*, along with a conjecture of a possible mapping between unique and multiplicative counts.

2. *Nodal betweenness centrality: Nodal betweenness centrality measures how often a node appears in the set of shortest paths (which we shall denote $\tau$), geodesics, of the network (Freeman, 1977). Nodes with high betweenness centrality will more frequently appear in shortest paths than low ranked nodes. The betweenness centrality of a node $n$ can be computed by*

$$B_n(n) = \sum_{s,t \neq n} \frac{s_{st}(n)}{s_{st}}, \tag{12}$$

where $s_{st}(n)$ denotes the number of shortest paths from $s$ to $t$ that contain node $n$. The removal of nodes with high betweenness centrality can significantly affect the flow of dynamical processes on the network (Albert et al., 2000).

3. *Connected component analysis: CCs (connected components) are sets of nodes where any node may be reached from any other node that is a member of the set. CCs are used to describe the macroscopic structure of a network, as opposed to clustering which describes the local, microscopic, distribution. Highly clustered networks contain many components that are weakly connected to each other by a few bridge edges. It has previously been shown (Green and Kiss, 2010) that the GCC (Giant Connected Component — the component that spans majority of the networks) of highly clustered networks are sensitive to edge removal such that removing even a low proportion of edges can be enough to isolate a large proportion of the nodes. In our analysis, we generate a list of all edges in a network, cycle through each edge and accept or remove it with probability $p$, compute the size and frequency of all components remaining, and plot the cumulative distribution of component size.*

4. *Motif frequency and distribution: Clustering (local or global) essentially measures the occurrence of triangles in a network. In its local definition it can measure the proportion two triangles that share an edge, neither can it describe loops of order-four or large. Thus the perspective of clustering is quite limited and by its very nature it is a very coarse measurement. In this paper all closed structures of order four (i.e. quadruples), open and closed, and complete squares motifs are considered at both network and node levels. It is possible to define new structure type metrics to cover structures larger than triangles by exceeding in preciseness to classic (third-order) clustering and limiting ourselves to 4-node structures connected in a loop. To this end we define four new structural measurements: the ratio of unclosed quadruples (open paths of 4 nodes, $o\phi_4^n$), cycles with a single diagonal ($d\phi_4^n$), and complete squares ($c\phi_4^n$) to all connected structures of 4 nodes. These quantities are used to measure the (i) global ratios of unique order-four structure counts to all unique path counts, closed and unclosed, (ii) the probability distribution of finding a structures of a certain type and frequency for each node. These measurements alongside clustering will provide a more complete picture of network organization. A brief analysis of how to compute non-trivial paths of length $l > 1$ is as follows (it may be that paths of length 0 refer to edges and paths refer to the number of edges and $A_{ij} = 1$ is the adjacency matrix):*

(a) Consider a path $P$ of length $l$, and identify a head $H(P)$ (1st node of the path) and a tail $T(P)$ (last node).
(b) For each neighbour $n$ of $T(P)$: if (i) $A(H(P), n) = 1$, (ii) it has not already been counted as a closed path and (iii) its reverse has not been counted as a closed path then count a closed path of length $l + 1$.
(c) For each neighbour $n$ of $T(P)$: if (i) $A(H(P), n) = 0$, (ii) it has not already been counted as an open path, and (iii) its
