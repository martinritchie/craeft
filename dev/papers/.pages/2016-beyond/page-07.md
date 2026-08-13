![Subgraph notation and position labeling diagram showing hypergraph structures. Subgraphs are labeled by G followed by a symbolic subscript for ease of reference.](figure)

**Fig. 1** Subgraph notation and position labeling. Subgraphs are labeled by $G$ followed by a symbolic subscript for ease of reference.

e.g., $G_\square$, the sum of the degree sequences of both types of hyperedge must also be equal. In practice, this can be achieved by generating a suitable degree sequence for one type of hyperedge and then randomly permute it to obtain a second sequence for the second hyperedge. $G_\square$ has two degree sequences, one for each hyperedge, and both must be equal. In this case, we generate the sequence and use it for both hyperedges.

The network generating algorithm will then form a dynamic list for each hyperedge, where a node with hyperedge degree $k$ will appear $k$ times. This is achieved by selecting nodes from the lists, at random and without replacement, and by following the subgraphs' hyperedge composition in order to construct subgraphs and the network. It is possible that self or multi-edges form in which case the selection is discarded and new samples chosen until a valid selection is made or the algorithm exits with a failure status.

In this paper we wish to both computationally generate networks and theoretically analyse them using ODEs. The local structure of the hypergraph degree distribution provides the link between theory and simulation. The construction of the PGF induced by the hyperedge distribution is key. The PGF is typically defined at the node level. At the simplest level nodes may belong to a number of subgraphs without further specifying the orbit or position within the subgraph (Vale et al. 2009). The PGF could be constructed at the level of hyperedge but would not differentiate between topologically equivalent positions in the subgraph, and this is what we are in our network generating algorithm (nodes may now be allocated asymmetric subgraphs Karrer and Newman 2010). Finally, the PGF can account for the positioning for all asymmetric subgraphs and this is the most detailed description that maps to a specific subgraph (used in the ODE derivation, Sect. 2.3). For network generation the PGF takes the general form,

$$\psi(z) = \sum_{\mathbf{k}} p_{\mathbf{k}} \prod_{i} z_i^{k_i},$$

where $\mathbf{k} = (k_1, k_2, \ldots)$ is a placeholder and $\tilde{\mathbf{k}} = (k_1, k_2, \ldots)$, $k_i \in \mathbb{N}_0$ denotes the number of $k_i$ hyperedges of type $i$, associated to a node. The symbolic form of the PGF provides more flexibility for computation. Let $G_0$ denote the ordinary hypergraph configuration model (Petri et al. 2014); this is a simple graph with $N$ nodes where the $k_i$ neighbours of $G_0$ are Poisson distributed with parameter $\tau_i$. The PGF of such a network is

$$\psi(z_1, z_2) = \exp\left(G_0(z_1 - 1) + G_0(z_2 - 1)\right) = \mathrm{e}^{\tau_1(z_1 - 1) + \tau_2(z_2 - 1)},
