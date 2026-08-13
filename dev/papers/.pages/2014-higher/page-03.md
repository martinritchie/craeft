needs to be included in the networks' construction. Rather than just lines, the number of lines and corners of motifs that originate from a node can be varied. In any given motif a node can be considered as a corner and the number of lines originating from a node that join it to other nodes in the motif, and hence the overall network, depends on the asymmetric structures. To generate a network using this method, the following steps are performed.

1. allocate to a node a number of stubs following a given degree distribution,
2. mathematically determine the configuration of corners and lines for each corner type,
3. count stubs for each corner type (where a node has a stub is a corner it will be part of a particular type) at a time removing stubs from a running corner list,
4. draw corners at random and without replacement from the running corner list and connect to other corners to form motifs,
5. repeat until all lists are empty.

Fig. 2 illustrates corner allocation for an example node. Due to the nature of the configuration model with loops and double loops may occur, these are allowed here. Since we are working with the constraint depending only on degree, the ratio of self and double

![Three example network topology diagrams. Left: 'Diamond' — a diamond-shaped motif with 4 nodes. Centre: 'Triangle 1' — a triangle motif plus a tail node. Right: '2-Squares' — two square motifs sharing an edge.](figure)

**Fig. 2.** MFI hyper-node configurations. The different topographic configuration of a homogeneous graph with $k = 3$ as edges are decomposed from local to global.

loops to network size becomes negligibly small in the limit of large networks (Newman, 2003).

In this paper homogeneous CM networks are used with $\langle k \rangle = 4$ and $\langle k^2 \rangle = 48$ and $\beta = 0.4$. The stub configuration is initialised from the degree distribution:

1. $p_k = 0.2$: with probability $p_k = 0.5$ the optimum of stubs is maintained based on independent links, and with probability $p_k = 1 - p_k$ the optimum is arranged into one complete square corner and one triangle corner.
2. $p_k = 0.8$: a node is allocated one complete square corner and one triangle corner.

For $p = 0.8$ the algorithmic does not allow overlaps between square and triangle stubs, ensuring causality is preserved.

Table 1 shows the expected motif allocation per node. The configuration model allows us to analytically determine some of the network structural properties, more specifically the PGF (Probability Generating Function) of the degree and clustering distributions.

The CM algorithm for this work is configured as follows. First, assign stubs to nodes in this study. Then $M_S$, choose the probability $p_s$ that each type of motif is used. In this work we chose one complete square and one triangle and $p_s$ two empty squares and one triangle. Parameters are chosen such that $\langle k \rangle = 3.4$ and the stub configuration is initialised from: $s_1$ (empty stub, i.e. a simple link), $s_2$ (single node), $s_3$ (triangle), $s_4$ (complete square), $s_5$ (empty squares), allowing nodes to be part of the following motif types: $s_3$ (triangle), $s_4$ (complete square), $s_5$ (empty square);

$$\Psi(x, y, z, s) = q_1 x + q_2 x^2 y^2 + q_3 x^3 y^3 z + q_4 x^4 s^4$$
(1)

and the original stub distribution may be recovered by substituting each $s_i$ with 1, where $N_s^{(k)}$ denotes stub cardinality:

$$g(x, y, z, s) = q_1 x + q_2 x^2 y^2 + q_3 x^3 y^3 z + q_4 x^4 s^4$$
(2)

$$= q_1 x + q_2 x^2 y^2 + q_3 x^3 + q_4 x^4$$
(3)

This yields 2.3 nodes, triangles and complete squares per node for each level of clustering used.

**Table 1**
The expected number of lines, triangles and complete squares per node for each level of clustering used.

| | Triangles | Complete squares |
|---|---|---|
| $p_k = 0.5$ | 0.5 | 0.5 |
| $p_k = 0.8$ | 1 | 1 |

![Cornerstone diagram. A node is initially allocated a number of stubs (here, $k = 5$). With probability $p_s = 1 - p_t$ the set of different structures as shown are possible and if $k$ is even there are $(k/2, 0, 0)$, i.e. the node is part of $(k/2)$ squares; the configuration of motifs will be adapted accordingly.](figure)

**Fig. 3.** Cornerstone diagram. A node is initially allocated a number of stubs (here, $k = 5$ lines). With probability $(p_s, p_t)$ the set of different structures as shown are possible and if $k$ is even there are $(k/2, 0, 0)$, i.e. the node is part of $\lfloor k/2 \rfloor$ squares; the configuration of motifs will be adapted accordingly.
