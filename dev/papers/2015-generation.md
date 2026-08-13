# Generation and analysis of networks with a prescribed degree sequence and subgraph family: higher-order structure matters

*Journal of Complex Networks* (2017) **5**, 1–31
doi:10.1093/comnet/cnw006
Advance Access Publication on 16 May 2016

Martin Ritchie

*School of Mathematical and Physical Sciences, Department of Mathematics, University of Sussex, Falmer, Brighton BN1 9QM, UK*

Luc Berthouze

*Centre for Computational Neuroscience and Robotics, University of Sussex, Falmer, Brighton BN1 9QH, UK*

and

Istvan Z. Kiss†

*School of Mathematical and Physical Sciences, Department of Mathematics, University of Sussex, Falmer, Brighton BN1 9QM, UK*

†Corresponding author. Email: I.Z.Kiss@sussex.ac.uk

Edited by: Ernesto Estrada

[Received on 26 November 2015; accepted on 9 March 2016]

Designing algorithms that generate networks with a given degree sequence while both varying subgraph composition and distribution of subgraphs around nodes is an important but challenging research problem. Current algorithms lack control of key network parameters, the ability to specify to what subgraphs a node belongs to come at a considerable complexity cost or, critically and sample from a limited ensemble of networks. To enable controlled investigations of the impact and role of subgraphs, especially for epidemics, we need algorithms that generate networks with a given degree sequence with the subgraph-level structural variants as diverse as possible. In this article, we present two new network generation algorithms that use subgraphs as building blocks to construct networks preserving a given degree sequence. Additionally, these algorithms provide control over clustering both at node and global level. In both cases, we show that being constrained by a degree sequence and global clustering results in topologies where degree and subgraph distributions and correlations are restricted. We suggest that for small- and large-scale network structure metrics such as path length and betweenness measures. Simulations of standard epidemic and complex contagion models on these networks show clearly that degree distributions and global clustering do not accurately predict the outcome of dynamical processes taking place on them. We conclude by discussing the benefits of the subgraph-centric approach.

*Keywords*: networks; clustering; subgraphs; epidemics; complex contagion.

## 1. Introduction

Being able to replicate, and therefore investigate, the structure and function of real-world complex networks is a profoundly difficult problem. However, the pervasiveness of a system that could be more



![Set of subgraphs G_0 through G_8 used in the article, illustrated as small graphs with labeled vertices and edges showing various triangle and edge configurations](figure)

FIG. 1. The set of subgraphs that have been used in this article. The subgraphs denoted by $\{G_0, G_1, G_2, G_3, G_4, G_5, G_6, G_7, G_8\}$ are those that have been used as input for the pairwise-network construction algorithm. We use $|\mathcal{G}|$ ($= |\mathcal{G}_3|$ in this article) to denote the total number of uniquely counted subgraphs given by the subgraph counting algorithm [12].

accurately interpreted as a result cannot be overstated: social networks [1], the spread of disease [2], artificial intelligence [3], language structure [4] and transportation networks [5]. Accordingly, a number of network models and network generating algorithms have been proposed [6–14]. Many of these network models seek to reproduce a specific network property or characteristic: the degree distribution [7, 15, 16], network motifs [17–19] or short-range correlations [5, 13, 15, 17]. Similarly, the triangle-count plays a role in a network [9, 20]. However, investigations of *higher-order* structure, subgraphs and arrangements of subgraphs not specific to a standard network model, have been limited by a lack of accurate and versatile network generation algorithms that embed higher-order structure, a shortcoming addressed by the work we seek to build upon.

In the standard configuration model, triangle subgraphs appear infrequently as a by-product of working with finite size networks [22]. But what if one wants triangle subgraphs to appear in a network, in particular, if one wants to model a complex network with clustering? An extension of the configuration model to this case exists [23, 24]. In this extension a node is allocated a number of stubs, that may go on to form standard edges, as well as a number of triangle 'corners' or *hyper-stubs*, pairs of stubs that will form triangles. Edges are formed in the usual way, triangles are formed by selecting three triangle hyperstubs at random and connecting their pairs and third element stubs.

In the number of all stubs must be divisible by two, the total number of triangle hyperstubs must be divisible by three is a necessary condition for the triangle hyperstub sequence to be graphical. Another similarity this model shares with the standard configuration model is that the probability that any two triangles will share an edge, thus forming a $G_8$ subgraph (see Fig. 1), is negligible for finite or large network size [21]. Just as a network comprised of lines only is limited in recreating real-world networks, so is a model that can only employ edges and triangles. Obviously, this may depend on properties and structure of the real networks, but in many cases, edges and triangles are not enough to produce an accurate enough artificial replica of the real network.


The configuration model has since received further attention to address this [21]. Building on the edge-triangle model, a more general subgraph-based approach is taken where one now specifies a distribution of edges alongside distributions of arbitrary subgraphs. In the case of complete subgraphs it is obvious how to do this. For example, $G_3$ subgraphs, say, can be formed by allocating to nodes hyperedges composed of three stubs. Then four of these hyperedges can be selected at random to form a $G_3$ subgraph. However, it is not clear how this may work for subgraphs that are composed of more than one type of hyperedge. For example, in a $G_5$, there are two different types of hyperedges, and it is necessary for any network model or construction algorithm to be able to make this distinction. A strict and nontrivial proposal that it is possible to identify a node's role within a subgraph using orbits. To find the orbits of a subgraph one must first list all possible automorphisms, where an automorphism is a permutation of the nodes of a subgraph such that all edges remain intact. The orbits are then the sets of nodes that can be permuted so that no edges are created or destroyed. Of course, computing the automorphism group of subgraphs is computationally challenging and doing so quickly remains an open problem [22].

Network models are rarely used independently of other processes. Instead, they typically provide the substrate for dynamical processes to operate upon. For example, the computational susceptible-infected-recovered (SIR) model of contagion is often embedded onto a network to help better understand how the network and its properties affect the epidemic. Previous work [13] successfully incorporated the Karrer and Newman approach into an approximation for the final epidemic size in an SIR epidemic. By sampling from the joint degree distribution, the authors ran Monte Carlo SIR simulations and showed excellent agreement with simulation results. In order to achieve this, Ritchie et al. bypassed the need to classify a node's role in a subgraph by using isomorphism instead. In their model, nodes' subgraph roles were uniquely enumerated, even if they were topologically equivalent to one another, and this enumeration defined their role. The motivation for this restriction was to simplify the derivation of the ODE model. Using the orbit approach or the full enumeration are different ways of satisfying different model constraints, and these are not the only possible approaches. In order to consider multiple equivalent nodes within subgraphs, one can instead classify nodes by the sub-cardinality of their hyperedges.

A common method to sum all of the above ideas is, i.e., mean-fields, to derive general degree-sequence-based measures. The degree sequence of a network, $\{k_i\}$, lists the number of connections for each and every subgraph that is to be included. From these sequences it is possible to recover the network's degree sequence, and to analyse this then by the sub-cardinality of the hyperedges which they represent and then summing the resulting sequences. Therefore the degree sequence of the network is a result of the construction of the subgraphs from a given sequence, and is outlined in Procedure 3b, given that the degree sequence of the network is probably the single most important characteristic of a network, there is a need for methods that can generate networks that are drawn from a specific distribution and yet preserve a given degree sequence. In [13], we recently showed that it is possible to constrain the subgraph sequences so that the 1st and 2nd moments of the resulting degree distribution match those of a target distribution. In the next section we review our algorithm and show how it can be used to construct the degree sequence and clustering.

The article is organised as follows. In Section 2, we describe in detail the two generation algorithms, including tuning of clustering. In Section 3, we validate our algorithm and we explore the diversity of the generated networks by comparing them to the widely used Erdős-Rényi scheme. We further analyse networks generated by using different subgraph families or distributions. Epidemic and complex processes on networks reveal that disease spreading depends on the degree distribution, and that clustering alone are not sufficient to predict the outcome of these processes. Finally, we discuss extensions and further research questions relating to our work.


## 2. Materials and methods

In this section we propose two new algorithms, both of which are parameterised by a degree sequence and a set of subgraphs. The algorithms construct hyperedral degree sequences (from which the input degree sequence may be recovered exactly) that can be used in a modified configuration model style connection procedure to realise a network.

The concept of a *hyperedge* underlying the input degree sequences are common to all configuration-like models. First it is necessary for a degree sequence to sum to an even number to be graphical. If it does not, a stub must be created or destroyed to satisfy this constraint. In general, hyperedge degree sequences must sum with multiplicity equal to the number of times they appear in their parent subgraphs, i.e., a triangle hyperedge sequence must be divisible by 3. When selecting stubs or hyperstubs at random to form subgraphs, it is possible that self or multi-edges may form. The number of these events (hyperlink depends only on the average degree $k$) and maximum subgraph size. As such, it may be possible to simply delete self-edges or collapse multi-edges down to a single edge. If this approach is taken then the graphing degree sequence will be retained. Invoking self and multi-edge deletion in the hyperedge degree approach corresponds to randomly selecting stubs, cancelling any connection violations, and repeating. This is known as the *matching algorithm* [25]. Finally, it is possible for the procedure to be left with no option other than to add self-edges or create existing links or otherwise violate the structure of the node. In this case we completely reset the algorithm, regenerating hyperstub sequences and forming subsequent connections until a network is formed.

### 2.1 The undetermined sampling algorithm

The concept underpinning this algorithm is that for each node there are combinations of hyperstubs that will satisfy its degree. For example, a node with $k = 2$ and self-edges could form from 1 hyperstub of $G_4$-edge and 1 $G_3$-hyperstub. The number of possible arrangements will depend on the degree of the node and number of input subgraphs. From these arrangements a single one is selected at random. For a given degree $k$, this problem is equivalent to solving an underdetermined linear Diophantine equation equal to $k$ subject to positivity constraints. The coefficients are given by the degrees of the input subgraphs, they are indexed by the input subgraphs, and the solution will give the number of each hyperstub so that the degree of the node is matched exactly.

More formally, let the degree of node $i$ be $k_i \in \mathbb{N}^+$, the input subgraphs be $\{d_1, d_2, \ldots, d_s\} \in \mathbb{N}^{s \times 1}$, and the set of subgraphs to be included in the network's construction, $\{G_1, G_2, \ldots, G_s\}$. The degree of $G_j$ is $d_j$, for example, the degree of a triangle is 2. Then the hyperstub vector for node $i$ is $h_i = \{h_{i,1}, h_{i,2}, \ldots, h_{i,s}\} \in \mathbb{N}^{1 \times s}$. From these arrangements we take a single vector that has the following linear underdetermined equation equal to $k$ (= 2), systematically list all its combinations as a single vector, we first concentrate all the hyperstub vectors into a single vector, so as to solve the following linear underdetermined equation:

$$a_1 x_1 + a_2 x_2 + \cdots + a_m x_m = b \tag{1}$$

where $k = a_1 x_1 + \ldots + a_{m \times 1}$ and $b$ denotes the number of eligible hyperstubs — a whole number in {1, ..., $b$} of this equation. A solution $x$ of this equation corresponds to the number of hyperstubs of each type that are assigned to the node. The vector $b/a_i$ corresponds to the number of each type of hyperstubs required to satisfy the constraint $a \cdot \frac{b}{a_i}$ = 1. A solution $x$ of this equation corresponds to the number of hyperstubs of each type for the given degree it can only go up to where the condition for the constraint $k \leq \frac{b}{a_i}$ is satisfied. The solution corresponds to the number of each type of hyperstubs required to satisfy the constraint $a \cdot \frac{b}{a_i} \leq 1$ A solution $x$ of this equation corresponds to the number of hyperstubs of each type for the given degree it can only go up to where the condition for the constraint $k \leq \frac{b}{a_i}$ is satisfied. [The remainder of this paragraph is cut off at the page boundary.]


if $a_i$ and $a_j$ take values 1 and 2 corresponding to hyperedges of $G_1$ and $G_2$, respectively), and the degree of the node $k \in S$ in the Diophantine equation would take the form $r = a_i \cdot d_k^i$, and the solution space of this equation is given by the pairs $(i_1, i_2) \in \{5/9\}, \{3, 1\}, \{1, 2\}$. These in general these equations may be solved recursively by fixing a trial value $a_i = j$ and reducing the dimensionality of the equation by absorbing this term. This is repeated until the equation becomes of the standard form $l = a_n \cdot v + u_{n,n}$, which can be solved explicitly. A solution obtained this way will form a single solution of the original equation. This process is then repeated for a different starting trial solution, and since we seek only positive solutions and $k$ is finite, the corresponding solution space may be finite number of elements. Matlab code for this process is available at https://github.com/martinritchie/Network-generation-algorithms. Accessed on 23 April 2016.

The VDA algorithm selects values for each subgraph configuration by iterating over the integer-valued hyperstub degree sequences. To proceed, the algorithm works sequentially through the degree sequence $D = \{d_1, d_2, \ldots, d_n\}$ of the $N$ nodes, where $d \equiv \{d_{i,m_1}, d_{i,m_2}, \ldots, d_{i,m_q}\}$, selecting a random solution from the solution space that corresponds to $k = d_i$, that specifies the hyperstub configuration, and then examining all the selected solutions for all the nodes, constituting a total partition of dimension $b \times N$, where $b$ denotes the total number of hyperedges induced by the input subgraphs, is formed.

For incomplete subgraphs it is not possible to select solutions of the Diophantine equations' solution spaces at random. The reason for this is two-fold: (1) not all incomplete subgraphs are composed of equal quantities of each of their constituent hyperedges, (2) the sum total of the degree of all boundary edges must equal that of all internal edges. Problem (1) may be solved because the number of hyperedges can be more readily accommodated into the degree of a node. Problem (1) may be addressed by representing every hyperedge induced by a subgraph as a partition and collecting equivalent partition of hyperedges by their sub-cardinality. Problem (2) may be addressed by decomposing hyperstubs generated in excess into simple/classical edges. It should be noted that this process can generate biased hyperedge sequences but that this bias is only present when incomplete subgraphs are specified as input for the Valiant–Dai algorithm (VDA). One advantage of this decomposition approach above is that it is possible to calculate the number of hyperstubs that will be decomposed back into stubs using integer partitions, and this is detailed in Appendix A.3. In particular the following holds true:

$$p(k, a) = \sum_{i} \mu(k, a, i) \cdot p(k - a \cdot i) \cdot p(k - a \cdot i + 1)$$

where $\mu(k, a, i)$ is the number of times that $a$ appears in the partitions of $k$, with $a \geq 1$.

This may be used to compute the number of times certain hyperstubs appear. Returning to the example of the homogeneous networks with $k = 3$, according to this formula there will be five counts of the double hyperstub generated for every two counts of the triple counter in the partition space, since 5 can be partitioned as

$$\{\{1, 1, 1, 1, 1\}, \{1, 1, 1, 2\}, \{1, 1, 3\}, \{1, 4\}, \{3, 2\}, \{2, 2, 1\}, \{1, 2, 2\}, \{2, 1, 2\}, \{5\}\}$$

and $p(5, 2) = 4$ and $p(5, 3) = 2$.. It should be noted that $p(k, a)$ will count how many times $a$ appears in *all* possible partitions of $k$, not just the partitions relevant for each value. Since it is not interested in finding a particular partition of each value, this simple number theoretic consideration allows each $a$ to be quickly quantified.

Pseudocode for the VDA algorithm is given in Appendix A.2, and the Matlab code is available from https://github.com/martinritchie/Network-generation-algorithms. Accessed on 23 April 2016.



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



To generate a CMA network, one needs to first decide on a degree sequence $D$, a subgraph set $G = \{G_1, G_2, \ldots\}$, and a set of subgraph sequences $S = \{S_1, S_2, \ldots\}$ where $S_i(k)$, with $j = 1, 2, \ldots$ and $k = 1, 2, \ldots, N$, gives the number of times a node will be part of a $G_j$ subgraph without specifying the precise hyperedges that connect to the node in $G_j$ subgraph. Our goal is to map the subgraph sequences onto precise hyperedge sequences that can then be allocated to nodes that can accommodate them. From the hyperedge sequence, it is possible to work out the lower bound on the degree of nodes that can accommodate a specific hyperedge sequence. To complete this mapping one needs to differentiate between complete and incomplete subgraphs.

For complete subgraphs the subgraph sequence is identical to its hyperedge sequence since there is only one way of a hyperedge by which a node can connect to a complete subgraph (by specifying the other nodes in the subgraph). Given the hyperedge sequence, it is straightforward to determine the minimum nodes that can accommodate the hyperedge sequence. For incomplete subgraphs the subgraph sequence does not specify how the node connects to the subgraph. Hence a mapping to subgraph to the various hyperedges are allocated to nodes. To see how to do this, let us consider an arbitrary subgraph $G$ with subgraph sequence $S$. Given that the subgraph has $p$ different edge types (edges $e_1, e_2, \ldots, e_p$), let the vector of probabilities of picking different hyperedges. We note that the values of $p$ reflects the proportion of each hyperedge found in the subgraph. For example, $G_3$ has two distinct hyperedges that both appear with multiplicity two, in this case $p = (1/2, 1/2)$. This will ensure that their numbers are balanced and subgraphs can be formed.

To map each subgraph sequence distribution to a hyperedge sequence, for each node $i$ with $S_j(i)$ as the subgraph sequence of index $j$ (this is not yet a node label), we pick hyperedge types to transform the subgraph sequence into hyperedge degree. For each $j$ this is done with a vector of length $p$ specifying the exact number of each hyperedge. It is possible to concatenate all the resulting choices from all multinomial distributions $M^{i}(C, p)$, where $i = 1, 2, \ldots, N$ from the following matrix

$$\begin{pmatrix} h_1^1(1) & h_2^1(1) & \cdots & h_{C_1}^1(1) \\ h_1^2(1) & h_2^2(1) & \cdots & h_{C_2}^2(1) \\ \vdots & \vdots & \ddots & \vdots \\ h_1^N(1) & h_2^N(1) & \cdots & h_{C_N}^N(1) \end{pmatrix} = H^1$$

where $h_j^i(1)$ denotes the number of $h_j$ hyperedges allocated to the subgraph degree $G_l$. We now need to consider the total number of edges specified by each column of the above matrix for each hyperedge type. This is given by $H^1 \cdot \mathbf{1}^T = (H_1^1, H_2^1, \ldots, H_{C_l}^1)$, where entries of edges are the total number of times edge $e_i$ appears in the network, i.e. the hyperedge sequence. Given the degree sequence $D$ and subgraph set $G$ and $i = 1, 2, \ldots, N$, this process needs to be repeated for each subgraph to be included in the network's construction, so that each node $i$ has the hyperedge sequence $P^i = (P_1^i, P_2^i, \ldots)$. There is a corresponding $H^{lk}$ with elements that the algorithm will use as the lower bound on the degree of the nodes that can accept such a selection of hyperedge types.

The algorithm then proceeds by choosing the largest values, $H_{max}$, from all $H^{lk}$ matrices, and this is used as a lower bound on the degrees of nodes that can accept the hyperedge configuration associated with $H_{max}$, i.e., have enough edges of the classical type. From the list of all nodes with degree equal to or larger than $H_{max}$, a node is selected uniformly at random. The degree of the selected node is reduced


accordingly, and the index of the node is now associated with the hyperédeb degree to $H_{m_n}$. This node is then removed from the pool of eligible nodes for that particular subgraph, as otherwise it may be selected twice for the same subgraph thus violating the subgraph degree sequence. Similarly, the element $H_{m_n}$ is also removed from the pool of subgraph degree sequence that have not yet been allocated. This process needs to be repeated until all elements of each subgraph degree sequence are allocated to nodes. Any edges that are not allocated to a particular hyperstub or subgraph are left to form edges.

In some cases it may be necessary to impose some cardinality constraints on the subgraph sequences. Obviously, if the network is homogeneous with $k = 3$ we cannot include complete pentagon subgraphs or allocate two $G_1$ subgraphs to each node. More generally, it may be necessary to constrain the moments of the subgraph sequence. Let $\langle k \rangle$ denote the mean degree of the graph degree sequence and let $G$ be a connected subgraph of the nodes of interest. If $\alpha(G)$ is the number of nodes in $G$ and $\sigma(G)$ the number of edges with mean $\langle s \rangle$ then $\langle s \rangle \leq \alpha(G) \langle k \rangle / 2$ is a necessary condition for the two sequences to be graphical. In the case of more than one hyperstub, this is extended to $\sum_b \langle s_b \rangle \leq \alpha_b \langle k \rangle / 2$ where $\alpha_b$ and $s_b$ denote the number of hyperstubs, hyperstub cardinality and associated subgraph sequence, respectively. For the networks generated in this article, the degree sequence and subgraph sequences were generated using networks previously generated by the UDA such that prior knowledge about the sequences being graphical was available without the need to impose any such constraints.

Clustering calculations for this algorithm are trivial since the subgraph degree sequences are known. One simply sums a sequence and then multiplies by two to get the number of triangles indicated by the triangular subgraph degree sequence which gives us the number of triangles in the network. The number of triples of connected nodes can be calculated following the method given for the UDA given in Section 2.1.1. Pseudocode for the CMA is given in Appendix B and the implementation code is freely available from https://github.com/martinitchie/Network-generation-algorithms. Accessed on 23 April 2016.

## 2.3 *Connection process*

We describe this process for a single incomplete subgraph. The case of the complete subgraph is trivial and has already been described earlier in detail. This process was first presented by Ritchie and Newman [21]. Consider a subgraph composed of three different hyperstub types, $h_1$, $h_2$ and $h_3$ that occur with a multiplicity of 3, 2 and 3, respectively, i.e., the subgraph is composed of six nodes. We require the following necessary conditions for the hyperstub sequences to be graphical

$$\sum_{i} d_i(h_j) = \sum_{k} d_k(h_j) \tag{3}$$

where $d_i(.)$ specifies the $h_j$ hyperstub degree of node $i$. If these conditions are not met, one needs to decompose any surplus hyperstubs into stubs that may be treated as excess nodes in the subgraph degree sequence.

Using the hyperstub sequences, one can create three dynamic lists for the three hyperstub types where a node appears with multiplicity equal to its hyperstub degree. Once the dynamic lists are fully populated, the connections process can start. This is done by sampling the following: 1 node from the $h_1$ bin, 2 from the $h_2$ bin and 3 from the $h_3$ bin, and all the selection processes done uniformly at random and without replacement. The selection process is done without replacement from the same bin to ensure that (1) the selection contains no duplicates (that will form self-edges) and (2) that no single pair of nodes are already connected. If a connection already exists, a multi-edge may form and/or



![Two plots (a) and (b) showing the average number of attempts to create a network and the average number of triangles and triangles by products, as a function of the input number of triangles. Plot (a) shows bars for degree 3, 4, 5 with the number of attempts on the y-axis and input number of triangles on the x-axis. Plot (b) shows similar bars with average number of triangles and triangles by products on the y-axis.](figure)

FIG. 2. (a) The average number of refusal attempts to create a network and (b) the average number of triangles found per network. In both cases the CMA was parametrized with only $G_0$ and $G_1$ subgraphs. $G_0$ subgraphs, $G_1$ are specified as input: both the number of average number of attempts and triangles by products decreases. Triangles by products are computed by subtracting the input from the total number of triangles.

subgraphs will share edges. If neither of these conditions are violated then the connections may be formed. Otherwise, all nodes are returned to the free node list and a new attempt is made. It is possible that after many selections no valid combinations of nodes remain. For example, all bins may contain the same node. In this and other edge cases, all bins are re-populated and the selection procedure is re-started anew.

As previously discussed, it is possible to delete self and multi-edges but this will destroy the degree sequence. The method of selecting nodes has been previously introduced and is known as the matching algorithm [23]. However, it has previously been shown that the matching algorithm introduces a bias when constructing networks [20]. Ideally, when a self or multi-edge is formed one would want the whole connection process from scratch, the so-called *refusing algorithm*. This results in an unbiased sampling [20]. For the configuration model the number of such self and multi-edges depends on the first and second moments of the degree distribution [2]. As such, an unbiased configuration model approach may result in prohibitive running times as $n$ increases.

Currently, there are no analytical results regarding the probability of self or multi-edges as well as bias for the subgraph connection process. To help develop some understanding, we set up the following experiment: we generate a set of networks at varying values of $k$ (degree), $n$ (nodes), and $G_t$ (triangles). Initially the CMA is parametrized with no $G_1$ subgraphs and only $G_0$ subgraphs, subject to the configuration model for incoming degree value of $k = 2, 3, 4, 5, 6$ and then record the average number of attempts required before a network is produced as well as the average number of $G_1$ by-products. We then repeat this but with the CMA parametrized with increasing numbers of $G_1$ subgraphs, distributed so that a node is incident to at most one $G_1$ subgraph, and so on. Figure 2 illustrates that both the number of attempts required and $G_1$ by-products decrease as network size increases, as one would expect. It also reveals that these quantities decrease when the CMA is parametrized with increasing numbers of $G_1$ subgraphs. Regardless of degree: Since the number of attempts per networks is a function of the number of self and


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


equation (5). To compute the probability of finding a second edge between nodes $i$ and $j$ in the $G_s$ model, one must compound $(k_i - 1)(k_j - 1)/(3m_s - 1)$ with equation (5):

$$p(\delta_{ij}(2) > 1) = \frac{k_i(k_i - 1)(k_j - 1)}{(3m_s)^2}$$

summing this probability over all pairings of nodes and dividing by $2s$ to remove the double count, yielding

$$\lim_{s \to \infty} \frac{1}{2s} \sum_{i=1}^{N} \sum_{j \neq i} p(\delta_{ij}(2) > 1) = \frac{1}{2} \sum_{i=1}^{N} \sum_{j \neq i} \frac{k_i(k_i - 1)(k_j - 1)}{(3m_s)^2} \tag{6}$$

$$= \frac{1}{2(3m_s)^2} \sum_{i=1}^{N} k_i(k_i - 1) \sum_{j \neq i} (k_j - 1)$$

$$= \frac{1}{2} \left( \frac{\langle k_s^2 \rangle - \langle k_s \rangle}{\langle k_s \rangle} \right)^2,$$

where we have used

$$3m_s = \langle k_s \rangle N_s, \quad \langle k_s^2 \rangle = \frac{1}{N_s} \sum_{i=1}^{N_s} k_i^2 = \frac{1}{N} \sum_{i=1}^{N} k_i^2. \tag{7}$$

We again compare this value to that of the standard configuration model with the substitutions $2s \to k$, $\delta$ and $2(3m_s) \to 2m$ yielding

$$\frac{1}{2} \left( \frac{\langle k^2 \rangle - \langle k \rangle}{\langle k \rangle} \right)^2 = \frac{1}{2} \left( \frac{\langle k^2 \rangle - \langle k \rangle}{\langle k \rangle} \right)^2,$$

where the r.h.s. represents Newman's original estimate for multi-edges in the configuration model [2]. Now we consider scenario (b): selecting the same triplet of nodes twice resulting in two multi-edges. Consider the nodes $i$, $j$ and $l$ with $G_s$ degrees of $t$, $s$ and $k$, respectively. This triple of nodes are connected with probability

$$\lim_{s \to \infty} \mu_{i,j,l} = \lim_{s \to \infty} \left( \frac{t \cdot s \cdot k}{(3m_s - 1)} \right)^2 = \frac{t \cdot s \cdot k}{9m_s^2};$$

the probability of this triple being selected twice is approximately

$$\frac{t_i s_{ij} k_{il}(t_i - 1)(s_{ij} - 1)(k_{il} - 1)}{(3m_s)^4}$$


12 M. RITCHIE *ET AL.*

This probability can be summed over all triplets of nodes yielding

$$\frac{1}{3} \sum_{i} \sum_{j \neq i} \sum_{k \neq i,j} \frac{\frac{s_i(s_i-1)s_j(s_j-1)}{(3m_s)^2} \cdot \frac{s_i(s_i-1)}{(3m_s)^2} (s_j-1)}{\frac{1}{3!G(1,1)} \left( \frac{\langle G'_2 \rangle - \langle G_2 \rangle}{\langle G_2 \rangle} \right)^2}$$

$$= \frac{1}{3!G(1,1)} \left( \frac{\langle G'_2 \rangle - \langle G_2 \rangle}{\langle G_2 \rangle} \right)^2 \tag{8}$$

where we have again used equation (7). The expected number of multi-edges created by two $G_s$-subgraphs connected on the same triplet of nodes is not constant with network size but instead tends to zero with increasing network size. This result, alongside equation (6), suggests that the number multi-edges in the $G_s$ model will be less than what is found in the equivalent configuration model network. We next consider the number of self-edges in the $G_s$ model network.

**The number of self-edges:** During the connection process of the configuration model self-edges are created when two stubs that are incident to the same node are connected. The analogue of this in the $G_s$ model is selecting then hyperstubs incident to the same node, resulting in three self-edges. We shall denote this event $\{i, i, i\}$.

$$p(\{i, i, i\}, \beta) \approx \frac{\binom{s_i}{3m_s-1}}{\binom{3m_s-1}{2}}$$

$$\lim_{N \to \infty} p(\{i, i, i\}, \beta) \approx \frac{s_i(s_i-1)(s_i-2)}{6(3m_s)^3}$$

this value can be summed over all nodes to estimate the expected number of self-edges in the network

$$\sum_{i} \frac{s_i(s_i-1)(s_i-2)}{6(3m_s)^3} = \frac{\langle G''_3 \rangle - 3\langle G''_2 \rangle + 2\langle G_3 \rangle}{6 V \langle G_3 \rangle^3} \tag{9}$$

where we have used equation (7). This value, like equation (8) is not fixed with network size and instead tends to zero as N becomes large.

**Node duplicates:** In the $G_s$ model, it is possible to select a pair of hyperstubs incident to the same node alongside a distinct third node, resulting in a self- and multi-edge. We shall denote this event $\{i, i, j\}$. Then

$$\lim_{N \to \infty} p(\{i, i, j\}, \beta) = \lim_{N \to \infty} \left( \frac{\binom{s_i}{3m_s-1}}{\binom{3m_s-1}{1}} \cdot \frac{s_j}{3m_s-2} \right)$$

$$= \frac{s_i(s_i-1)}{2(3m_s)^2} \tag{10}$$


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



![Three unconnected subgraphs with nodes arranged in triangular and polygonal structures, illustrating the subgraph generation constraints described in Section 2.2; the subgraph of triangle (A,B,C) and triangle (D,E,F) results in three annotated distinct tours (A,B,C,D), (B,C,E) and (D,C,F) overlapping on one unintended triangle (C,F,D).](figure)

FIG. 3. Unintended generation of subgraphs with overlap. Despite satisfying the generation constraints given in Section 2.2, the subgraph of triangle $(A,B,C)$ and triangle $(D,E,F)$ results in three unintended distinct tours $(A,B,C,D)$, $(B,C,E)$ and $(D,C,F)$ overlapping on one unintended triangle $(C,F,D)$.

## 2.5. *Models of contagion*

In order to illustrate the impact of network structure—and higher-order structure particularly—different epidemic dynamics were simulated on the generated networks. Three different models were chosen: susceptible-infected-susceptible (SIS), SIR and complex contagion [30, 31]. To simulate SIS and SIR dynamics, the fully susceptible network of nodes is perturbed by infecting a small number of nodes. Infected nodes spread the infection to susceptible neighbours at a per link rate of infection $\tau$. Infected nodes either recover and become susceptible again (for SIS epidemics) at per node rate of recovery $\gamma$ or become removed (for SIR epidemics). In contrast to the infection process in the previous two dynamics, the complex contagion process requires that susceptible nodes are exposed to multiple infection events before becoming infected. These events must be from different infectious neighbours as only the first infection attempt from an infectious node counts. This critical infection threshold for each node is set in advance and is usually bounded from above by the degree of the node. To simulate the complex contagion dynamics, nodes are allocated infection thresholds $r_i \in \mathbb{N}$, where $i = 1, 2, \ldots, N$, and the fully susceptible network of nodes is then perturbed by infecting a small number of initial nodes. Under this model a susceptible node $i$ becomes infected as soon as it has received at least $r_i$ infectious contacts from $r_i$ distinct infected neighbours. There is no recovery in this model and infected individuals remain infected for the duration of the epidemic.

## 3. Results

### 3.1 *Algorithm validation*

To validate our algorithms, we generated a number of networks with pre-specified degree distribution and subgraph set, as well as a multinomial distribution of subgraph corners or hyperstubs around nodes. We verified that the networks generated were as expected given the input.


![Four small networks (a), (b), (c), (d) generated by the Big-V, UDA, and CMA algorithms, each showing nodes and edges with varying topology](figure)

Fig. 4. Small networks generated by the Big-V, UDA and CMA algorithms. All networks have the same homogeneous degree sequence with $k = 7$. The Big-V algorithm created the random network, Fig. 4a. The UDA was parametrised with subgraphs $G_1$ (triangles) and $G_2$ (squares), with $C = 0.12$ and $C = 0.32$, respectively, to create Figs. 4b and 4c. All three of these networks have a global clustering coefficient of $C = 0.32$. The network nodes are coloured so that lighter/darker grey denote nodes of low/medium/high clustering, respectively. (a) Random, (b) Big-V, $C = 0.12$, (c) UDA, $C = 0.12$, (d) UDA, $C = 0.32$.

As described in Section 2, the algorithms preserve the degree sequence, permitting at most a single edge to be deleted if the degree sequence sums to an odd number. The ability to exercise control over the networks' subgraph topology is illustrated by Fig. 4. Note that Fig. 3a shows a random network that includes $G_1$ subgraphs. When constructing networks using the configuration model it is possible to create $G_1$ subgraphs with non-zero probability and this is to be expected [32]. However, this is a function of mean degree not network size, and this probability goes to zero with network size going to infinity.



TABLE 5. *Subgraph counts for the networks of Fig. 4. Note: if one adds a single $G_1$ so that it shares a single edge with a $G_0$ and this edge is not the diagonal edge of $G_0$, then $\Delta$ increases by one but this will have only increased by one, not two. We note that $2 \cdot \Delta$ yields the maximum number of possible $G_1$ induced by $G_0$. In general, calculating the number of $G_1$ in this way will always yield the maximum possible count but not necessarily the true count because a single $G_1$ could be shared by more than one $G_0$.*

| | $K_3$ | $P_3$ | $G_0$ | $G_1$ | $G_2$ | $G_3$ | $G_4$ |
|---|---|---|---|---|---|---|---|
| Random | 0 | 0 | 42 | 17 | 448 | 8 | 682 | 1706 |
| CMA | 1 | 4 | 31 | 8 | 246 | 5 | 429 | 1063 |
| FDA | 7 | 10 | 22 | 3 | 245 | 1 | 388 | 1219 |
| CMA | 0 | 9 | 20 | 40 | 185 | 24 | 389 | 1226 |

To properly demonstrate the proposed algorithms' control over the building blocks in the network, we used a recently proposed subgraph counting algorithm [12] to count the number of subgraphs *a posteriori*. In our implementation, we counted subgraphs composed of four nodes or less—or the top two rows of Fig. 1, so $K_3$ at 5- and 6-cycles. Table 5 gives the subgraph counts for the networks displayed in Fig. 4. It confirms that the random network given in Fig. 4a contains $G_0$, created uniquely as described above. The table also reveals that, in this instance, the random network is free of triangles, in agreement with the algorithms' design. The counts for $G_1$ subgraphs in Table 5 also confirm a significant presence of these subgraphs when compared with the random network. Although the CMA was parameterised solely with $G_1$ subgraphs, meaning we can see that each motif is 2-4 $G_0$ subgraphs, the subgraph counts reveal that this network contains 9 $G_0$ subgraphs. This is a consequence of attempting to generate small networks with such a high prevalence of triangles: it is highly likely that the algorithms will select nodes that already share one other common neighbour later in the connection process. The excess, the proportion of those events is become increasingly negligible with greater network size.

Next, we used the above motif counting algorithm to evaluate the extent to which the proposed algorithms maintain control over the prevalence of subgraphs in the generated networks. Figure 5 compares measured counts of subgraphs in LDA and CMA networks with expected counts. Here, an expected distribution would be the same as the actual distribution. We note that $G_0$ and $G_1$ appear in significant quantities: 33, 100 and 333 times, respectively, and regardless of network size. They are a natural consequence of the LDA as the probability of selecting two nodes in different branches of a finite tree-like network is non-zero. Therefore, our expected counts are the sum of the counts expected to occur organically (i.e. observed in the random networks). For example, since the CMA networks were generated with each node being incident to a single $G_0$ subgraph, a total of 833 unique occurrences of subgraphs were expected. Similarly, counting the number of $G_1$ subgraphs: since we are generating a network so that each $G_0$ is incident to two $G_1$ subgraphs, for a network of size $N = 5000$, our expected count was $833 + 344 = 1177$. The measured count was found to be 1165. More generally, we found the expected counts agreed to 97% with the observed counts, indicating that the generating algorithms did not create by-products in addition to those observed at random.<sup>7</sup>

<sup>7</sup> Although we will show in Section 7.1 that for specific parameterisations of CMA, by-products are possible.


![Bar charts comparing subgraph counts in CMA networks versus random networks, for subgraph m=4 (left) and subgraph m=5 (right), across increasing network sizes](figure)

FIG. 5. A comparison of subgraphs found in the UIA and CMA networks to their random network analogues and expected counts plotted with thick lines, thin lines and dashed lines, respectively. *cf* and *lll* denote the counts of $G_1$ and $G_2$ respectively. All networks have the same homogeneous degree sequence with $k = 5$ but with increasing size $N \in \{50, 500, 1000, 2500, 5000\}$. The random 100 of each size was generated. The UIA algorithm is parametrized with subgraph $\{G_0, G_1, G_2, G_3\}$ and the CMA with a single $G_1$ and $G_2$ subgraph, and the resulting average subgraph counts and the closing are shown on the right. The expected values were calculated by summing the total count from the subgraphs and the number of times each subgraph is potentially combinable, and adding these figures to the number of subgraphs found in by products in the random networks.

However, these results also suggest that the level of control exerted by the algorithms over subgraph prevalence depends on how often those subgraphs appear as by-products. Control is strongest for subgraphs that do not appear naturally as by-products. When considering subgraphs that appear naturally with high frequency, e.g., $G_{12}$, real control over their prevalence can only be achieved with even higher frequency is imposed, which may not always be possible for a given degree sequence and global clustering.

In what follows, we set out to highlight differences between the new algorithms compared to classic ones and also to emphasize the diversity within networks generated by the same algorithm.

## 3.2 Sampling from a different area of the network state space

In this section, we seek to highlight the versatility of the proposed generation mechanisms by showing that, given a degree distribution and a global clustering, they sample different areas of the network state space by targeting different subgraphs. The standard CMA algorithm creates triangles by searching for paths of five nodes and rewires each paths so that additional triangles are created. In other words, the principal building block of this algorithm is the $G_1$ subgraph and subgraphs that may be constructed by overlapping $G_1$ subgraphs. It follows that this algorithm is unlikely to give rise to a higher than expected at random number of $G_4$ or other 'empty' cycles. The UIA was therefore parametrized with subgraph family $\{G_0, G_2, G_1, G_2, G_3\}$. In order to eliminate the effect of degree heterogeneity, a homogeneous degree sequence with $k = 5$ was used. The resulting network had a global clustering coefficient of $C = 0.04$, induced by 666 (uniquely counted) $G_4$ subgraphs. We then used the Big-V algorithm to rewire random networks constructed using the same degree sequence until the desired level of clustering,


![Plots of average path length and diameter for Nantengu type networks (N = 5000 and k = 5) versus local family h. The Big-V algorithm was parameterised solely by clustering, in this case C = 0.05, to best suit the networks produced by the UDA, with remaining parameters varying by local family. The CMA was parameterised to reproduce the same degree sequence and global clustering. For each family we generated 15 networks from each algorithm, together with the target networks; three target networks and three from each algorithm. Three families were significant: (a) Average path length, (b) average betweenness and (c) maximum betweenness.](figure)

$C = 0.04$, was achieved. Significant differences between generated networks would confirm that the Big-V and UDA generated networks are sampled from different areas of the state space of networks satisfying that degree sequence and global clustering. As a further point of reference, data taken from a random network realisation, generated using the configuration model of the degree sequence, are also shown in all of our analyses. Henceforth we shall refer to these three types of networks as network family *h*.

In Fig. 6, the distributions of the average path length, average betweenness centrality and maximum betweenness centrality for the above networks are given. In general, an increase in clustering results in a higher value of the average path length — this is seen across all depth of random and Big-V networks in Fig. 6a. This is a known result [11]. Surprisingly, a similar magnitude of difference in average path length and average and maximum betweenness centrality is seen for the worst the Big-V networks versus the networks generated by the UDA. The results from generating networks using the UDA subgraph counting algorithm (Fig. 7) confirms that, as expected, the Big-V algorithm does not generate more $G_2$ subgraphs than are observed in the random network. More generally, the results show that the Big-V and UDA networks exhibit markedly different subgraph topologies with the Big-V networks relying heavily on $G_6$ to cluster the networks unlike UDA networks that rely almost exclusively on $G_1$, not appearing as part of any other subgraph. It may be that such variation was facilitated by the low level of clustering considered, and that with higher clustering, eliciting such differences might be more challenging. However, these results provide evidence that the UDA can sample from a different part of the state space than the Big-V algorithm.

## 3.3 *Diversity within the newly proposed algorithms*

In this section, we illustrate the diversity of networks generated with UDA and CMA by exploring the impact of subgraph distribution over nodes (the identical degree distribution and global properties) and how it may change network characteristics.

To do this we first parameterised the UDA with subgraph family $\{G_0, G_1, G_2, G_3, G_4\}$ (chosen due to its frequent use in the literature, e.g. [11–13, 21, 33, 34]), and a heterogeneous degree sequence generated from the Poisson distribution with $\lambda = 5$. Since it is difficult to count a priori how many copies that appear in a network generated using the UDA, we counted the total number of each subgraph, from UDA-produced subgraph sequences, and used these counts to create alternative subgraph sequences as


![Bar charts showing distributions of number of subgraphs (cd and cd8) and count for Network Random Elog-V/UDA](figure)

FIG. 7. Distributions of total number of subgraphs in networks formed by the CMA. In Fig. 6 and CDA networks have a global clustering coefficient of $C = 0.04$. All given counts are unique. The $G$ counts denote the number of $G_{ij}$ subgraphs that are not involved in any subgraph of type $G_{ij+1}$. The subgraphs shared by $G_3$ and $G_4$. The number of $G_{ij}$ subgraphs generated by the Reg $N$ algorithm is very close to the counts below the graphs $G_3$ and $G_4$.

input to the CMA, see Section 2.2, rather than drawing such sequences from a theoretical distribution. The resulting networks were therefore expected to have identical degree sequence, global clustering of 0.13 and subgraph counts. Since the CMA allows us to choose arbitrary sequences of subgraphs, we opted to push the clustered subgraphs, $\{G_1, G_2, G_3\}$, aside to manage the effect of clustering. We did this by specifying that these subgraphs had to appear with multiplicity greater than one. For example, a degree-three $G_3$ hyperstub required a minimum $k = 9$-degree node. As previously, we included a random network realisation of the heterogeneous degree sequence for comparison. Henceforth, we shall refer to these three types of networks as network family **B**.

The heterogeneity in degree distribution allows us to use additional degree-dependent metrics: degree-degree correlations and degree-dependent clustering, shown in Fig. 8. The plot for the degree-degree correlation coefficient shows that by aggregating clustered subgraphs around high-degree nodes, the CMA-constructed networks yield a higher assortativity than that of CDA and random networks, see Fig. 8a. This is an important property of the methodology, since the clustering potential of a network is bounded by the degree-degree correlation in heterogeneous networks. Moreover, if one wishes to maximise clustering in heterogeneous networks, it is necessary for nodes of similar degree to mix preferentially. Figure 8b shows that the CMA networks yield a negatively skewed distribution of degree-dependent clustering, with nodes of degree $k \geq 9$ contributing most to clustering. The ability to manipulate the degree and clustering relationship as well as assortativity clearly demonstrates the


20 M. RITCHIE *ET AL.*

![Two panel plot: (a) shows assortativity vs network type (Random, UDA, CMA) with lines for Network, CMA, UDA; (b) shows local clustering vs degree for the same network types](figure)

FIG. 8. Plots of assortativity and degree-dependent average local clustering for network family **B** with $k = \text{Poi}(5)$. The UDA and CMA networks have a global clustering coefficient of $C = 0.13$. The distribution of subgraphs in CMA networks was adjustable so that the subgraphs $\{S_1, S_2, S_3\}$ appeared with frequencies $\{0.675, 0.225, 0.1\}$ (approximately) rather than equal frequencies, so as to preserve the global clustering coefficient and a more positively skewed distribution of degree-dependent clustering. (a) Assortativity and (b) degree-dependent clustering.

![Two panel plot: (a) shows average path length vs network type (Random, UDA, CMA) with lines for Network, CMA, UDA; (b) shows diameter vs network type (Random, UDA, CMA)](figure)

FIG. 9. Plots of average path length and diameter for network family **B** with $k = \text{Poi}(5)$. The UDA and CMA networks have a global clustering coefficient of $C = 0.13$. The similar increase between UDA and CMA networks is a reflection of the higher assortativity of the CMA networks. The similar increase between UDA and CMA networks is a reflection of the higher assortativity. (a) Average path length and (b) diameter.

broader scope of the CMA when sampling from the ensemble of networks with same degree distribution and global clustering.

As with network family **A**, an increase in average path length, diameter, average and maximum betweenness centrality of CMA networks over random networks will be attributable to the higher global clustering coefficient, $C = 0.13$, see Figs. 9 and 10. However, since UDA and CMA networks share the same degree sequence and global clustering coefficient differences in these metrics between UDA and CMA can only be due to increased degree–degree correlation and negatively skewed


![Bar charts comparing clustering coefficient for network family B and C, showing minimum, average and maximum values for UBA and CMA networks across subgraph families](figure)

FIG. 10. Plots of betweenness centrality for network family **B** and **C**. The UBA and CMA networks have a global clustering coefficient of $C = 0.13$. A trend of increasing average and maximum betweenness centrality is observed between turning 5% and CMA networks, respectively. (a) Minimum, (b) average and (c) maximum betweenness centrality.

distribution of degree-dependent clustering. It has previously been noted that increased assortativity corresponds to an increase in average path length [35], and this will be compounded by the higher-degree nodes observed in networks generated using CMA. The resulting increase in clustering (both overall and as a function of path length) will be due to these highly clustered high-degree nodes. Finally, Figs. 10b and c show a significant increase in average and maximum betweenness centrality for networks in family **B** and **C** networks. This is yet another manifestation of the presence of these highly clustered high-degree nodes.

Table 2 presents a comparison between measures taken across the average subgraph (aggregated subgraphs in family **B**). As before, there is good agreement for UBA networks; it is observed that UMA networks have produced an artefact other than those in the intended distribution, e.g., additional 4-cycles have appeared as by-products. The effects of finite size have been exacerbated by aggregating clustered subgraphs around higher-degree nodes, effectively providing a route to medium degree nodes that are not directly connected to the hub. Additionally, the construction of subgraph families has highlighted a key challenge: when adding only a single edge may create additional (unwanted) subgraphs. This highlights the fact that while the total number of $G_s$ is preserved (as evidenced by identical global clustering) the way these subgraphs contribute to higher-order structure can vary significantly.

This section has highlighted that control over the choice of subgraph families and their distributions makes it possible to flexibly explore the solution space of networks with the same degree distribution and global clustering. This in turn provides us with the means to investigate specific areas of this solution space as well as further our understanding of how network metrics deal with such diversity.

## 3.4 *Does higher-order structure matter?*

In order to answer this question we make use of the network families **A** and **B** detailed above and test the impact of higher-order structure by considering the outcome and evolution of widely used dynamics on networks; namely, *SIS*, *SIR* and the complex contagion model.

For each network type in families **A** and **B** a series of networks, were generated. For each network, we performed a single Gillespie realisation of the relevant dynamics to completion (or until *t* = 2000). The evolution of infectious prevalence was then calculated, plotted and compared between network types. The SIS infection dynamics was simulated in a single-seed infectious trajectory and considering that a single infectious contact was usually not sufficient to result in an infected node. Different thresholds of infection and infectious seeds were used and these are specified in figure captions. Matlab code for the



TABLE 2. *Subgraph counts for network B* ($N = 5000, \lambda \sim \text{Pois}(5)$ *and* $\xi = 0.15$)*. The counts are unique. The expected counts are computed by summing the total counts from the subgraph sequences, dividing them by the subgraphs' node cardinality, and adding these figures to the number of subgraphs found as k-products in the random network. These are then subtracted by subgraphs that do not appear in any other subgraph.*

|          | $c4$ | $d4$ | $e4$ | $\langle 7 \rangle$ |
|----------|------|------|------|---------------------|
| Random   | 0    | 0    | 0    | 0                   |
| SDA      | 243  | 504  | 987  | 718                 |
| CMA      | 232  | 541  | 772  | 691                 |
| Expected | 243  | 504  | 619  | 741                 |

![Two plots side by side. Plot (a) shows SIS epidemic dynamics over time with a bell-shaped curve of infected nodes peaking around 4000 near time 40, with random network shown as dotted line and triangle markers, SDA as solid line with circle markers. Plot (b) shows SIR epidemic dynamics over time with a similar bell-shaped curve of infected nodes peaking around 1000 near time 15, then declining to zero, with the same marker conventions.](figure)

FIG. 11. (a) SIS and (b) SIR epidemic dynamics for network family B. The random, big-V and CDA lines have been plotted with a solid line, a circle and triangle markers, respectively. The SIS and SIR counts shown are the average of single Gillespie simulations on networks over 200 unique random, 200 SDA and 200 CMA networks. Both simulations were run using a single infected node as initial infectious seed of $I_0 = 10$ and had a per link rate of infection of $\tau = 1$ and recovered independently at rate $\gamma = 1$.

SIS and SIR Gillespie algorithms is available from https://github.com/martinritchie/dynamics. Accessed on 23 April 2016.

We know by construction that members of network family A were generated using different subgraphs, and Section 3.1 has shown that observable differences were found between networks in terms of average path length, betweenness centrality and subgraph composition. Despite this, Fig. 11, which show the time evolution for SIS and SIR dynamics, respectively, indicate that these dynamics can display a certain degree of insensitivity to these differences in structure. In this case, it is the SIR dynamics that show the greatest difference, in peak infectious prevalence (Fig. 11b) albeit quite marginal.



![Two panels showing complex contagion dynamics for network family A. Left panel: probability over time (x-axis 10–50) for Random, Big-V, and UDA networks. Right panel: final size probability distribution (x-axis 1000–5000) for the same three network types.](figure)

FIG. 12. Complex contagion dynamics for network family **A**. The complex contagion epidemics we parameterised as initial infections seed of $I_0 = 250$ and a final threshold of infection of $r = 2$.

![Two panels showing SIS and SIR epidemic dynamics for network family B. Left panel: number infected over time (x-axis 0–6) for Random, UDA, and CMA networks, showing curves peaking around time 2–3. Right panel: similar dynamics over time (x-axis 0–500) showing final epidemic trajectories.](figure)

FIG. 13. (a) SIS and (b) SIR epidemic dynamics for network family **B**. The random, UDA and CMA data has been plotted with a thin line to show the spread of epidemic trajectories. The bold lines show the mean trajectory based on each of the 1000 network realisations from each network generation algorithm. The SIS and SIR epidemics were seeded with an initial infection seed of $I_0 = 10$ and had a per link rate of infection of $\beta = 1$ and recovered independently at rate $\gamma = 1$.

In contrast, complex contagion dynamics do show sensitivity to structural differences found between Big-V and UDA networks. Figure 12 reveals that for UDA networks the epidemic fully percolates in almost 100% of the simulations instead of only 80% of the cases for Big-V networks and that epidemics on UDA networks achieve this steady state in less time. This indicates that whilst UDA networks operate in the super critical regime, Big-V networks are closer to the transition point. Locating this transition is possible but is beyond the scope of this article.

When network family **A** is used, the networks' degree distribution and clustering appear to be the main determinants of the time evolution and outcome of the SIS and SIR epidemics. In contrast, when network family **B** is used, Figs. 13 and 14 show that all dynamics considered are impacted by differences in network



![Complex contagion dynamics plot showing probability over time and final sizing for Random, UDA, and CMA network types](figure)

Fig. 14. Complex contagion dynamics for network family **B**: the complex contagion algorithm had an initial seed of the epidemic with a fixed threshold of infection of 2.

topology. For Figs. 13a and b, a trend of inhibited spread of infection is observed from the random to UDA to CMA networks. It has already been shown that clustering slows the spread of infection [5, 6], and we see that this effect dominates over higher assortativity, which usually leads to faster initial spread of the epidemic [37]. Similarly, Fig. 14, which shows the distribution of the final epidemic size for the complex contagion dynamics, reveals that (a) the higher clustering observed in the UDA networks fails to have a significant impact when compared with random networks, and (b) the CMA networks are significantly slow the pace of the epidemic as well as reduce its final size compared with both random and UDA networks. Given that the UDA and CMA algorithms target degree sequence and global clustering are identical the observed differences are explained by the combined effect of varying distributions of subgraph around nodes and varying prevalence of subgraphs (both of which are related to the local clustering coefficient distribution in the network).

Taken together, our simulation data show that even though the proposed algorithms construct networks with identical degree sequence and global clustering, these networks can give rise to measurable differences in resulting epidemics, be it in time evolution or final outcome. With the exception of *SIS* and *SIR* epidemics on network family **A** (still with some small differences), we found significant differences in all other instances. A more systematic investigation of more network models and wider parameter range for the dynamics is needed but is left to future work.

## 4. Discussion

In this article, we have described two novel network generating algorithms that strictly preserve a given degree sequence whilst permitting control over the building blocks of the network and enabling the tuning of global clustering. We have compared these algorithms to one another as well as to the widely used Bay-Y rewiring algorithm. Using our algorithms we have empirically demonstrated that it is possible to create networks that are identical with respect to degree sequence and global clustering, yet elicit measurable differences in network metrics and in the outcome of dynamics simulations run on them. We have presented evidence to suggest that the methods sample from different areas of the network state space and that these sampling variations do matter.


of subgraph by-products can appear in addition to what was observed in the random networks depending on how one wishes to place the subgraphs around nodes.

We have seen that by using a modest selection of subgraphs, we have been able to substantially influence dynamics running on the network, particularly for SIR contagion dynamics. All results relative to this model indicate that constraining a network by degree sequence and clustering is not sufficient to accurately predict the outcome of the epidemic. More importantly, the results appear to suggest that the location of the critical regime depends on the higher-order structure of the network (above and beyond clustering).

We have also constructed networks with different numbers of prescribed triangles, which is certainly a key feature of any network construction algorithm. However, if such structural details do not impact on dynamics, their practical value is in doubt. By comparing models with similar structural properties with a limited set of network descriptors. Although degree sequence, degree–degree correlations and global clustering coefficient were observed to be the main drivers of disease transmission in models such as SIS and SIR, we found it not to be true in general. This is an important finding because one should remember that the dynamics simulated here are modest in complexity, when compared with models of neuronal dynamics for example, and yet, we were able to elicit significant differences by simply tuning the network structure above and beyond triangles. This implies that accounting for type and impact of higher-order structure may yet hold and reveal many important and surprising results.

## Acknowledgements

MR gratefully acknowledges Engineering and Physical Sciences Research Council (EPSRC, Doctoral Training Grant EP/K503198/1) and the University of Sussex for funding for his PhD. We would also like to thank Dr J.C. Miller for fruitful discussions on the pairwise contagion model [13], and for sharing his code for simulating the complex contagion model on networks [38].

## Appendix

### A.1 *Integer partitions*

The set of partitions of a positive integer $k$, lists all possible ways of writing $k$ as the sum of other positive integers. For example, the set of partitions of 4 is: $\{[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]\}$. The number of ways to partition an integer is given by the partition function. For this derivation only we use $p(k)$ to denote the partition function of a positive integer $k$ and note that $p(1) = 1$, $p(2) = 2$, $p(3) = 3$, $p(4) = 5$, $p(5) = 7$, etc. For $k = 1, 2, 3, 4, 5$ ... the partition function returns $p(k) = 1, 2, 3, 5, 7, 11, \ldots$ respectively, and by convention $p(0) = 1$ and $p(-k) = 0$. This function can be used to calculate how many times a value for an integer $n < k$ appears in the partitions of $k$. We first compute the number of partitions in which $n$ will appear at least once. To determine the partitions we write $k$ as a partition in the following way: $[k - n, n]$ and use this notation to enumerate each part $n$ that appears in all remaining partitions of $k - n$. For example:

$$\{k = 4, n = 2\}:$$

$$[k - n + n] = [4],$$
$$[k - n + n] = [2 + 2] = [2, 2],$$
$$[k - n + n] = [1 + 1 + 2] = [1, 1, 2],$$
$$[k - n + n] = [3 + 1] = [3, 1], \tag{A.1}$$


i.e., there will be $p(k - a)$ such partitions. To more formally show this we use Euler's partition theorem

$$\sum_{n=0}^{\infty} p(n) z^n = \prod_{k=1}^{\infty} \frac{1}{1-z^k}$$
(A.2)

To calculate values of $p$, we first expand the r.h.s of the above

$$(1 + z + z^2 + \cdots)(1 + z^2 + z^4 + \cdots)(1 + z^3 + z^6 + \cdots) \cdots$$

such that to find the value of, e.g., $p(2)$, we simply collect the powers of $z^2$ to reveal the coefficient

$$p(2)z^2 = 2z^2.$$

In general, the terms in the geometric series in powers of $k$, i.e., $(1 + (z^k)^1 + (z^k)^2 + \cdots)$, give the number of times the integer $k$ may contribute to $n$ for the $z^n$ term, assuming that $k \leq n$. For example, $z^5$ can be formed by $(z^1)(z^4)^1$, which means that $5 = 2 + 2 + 1$ or by $(z^1)(z^2)^2$ which means that $5 = 1 + 1 + 1 + 2$, or $(z^1)(z^2)^2$ which means that $5 = 2 + 2 + 1$. To prove that $p(k - a)$ gives the number of partitions which $a$ appears at least once, we write the following

$$(1 + z + z^2 + \cdots)(1 + z^2 + z^4 + \cdots) \cdots (1 + z^{a-1} + \cdots) \cdots$$

where the exclusion of the $z^0$ term guarantees that the power of each and every term includes at least one alpha. Then to express the terms to the power of $n$ as

$$z^a + z^{2a} + \cdots = \frac{1}{1 - z^a} - 1$$
$$= \frac{z^a}{1 - z^a}$$
(A.3)

Euler's theorem can be modified to account for such a term

$$z^a \sum_{n=0}^{\infty} p(n) z^n = z^a \prod_{k=1}^{\infty} \frac{1}{1-z^k}$$
(A.4)

Comparing like-for-like powers in this modified expression gives the coefficient of $z^n$ as $p(n - a)$. Similarly, by writing

$$z^{ma}(1 + z^a + z^{2a} + \cdots) = z^{ma} \frac{1}{1 - z^a}$$

the result holds for multiples of $a$: $p(n - ma)$, the number of partitions in which $a$ appears at least $m$ times. Using the cumulative property of this expression, it is possible to compute the number of partitions in which $a$ appears exactly $m$ times

$$p(k - a(m-1)) - p(k - am)$$


multiplying this by $m$ and summing over all multiples of $a$, $m \cdot ma \leq k$ will give the number of times that $a$ appears in the partitions of $k$

$$p(l, a) = \sum_{m=1}^{\lfloor k/a \rfloor} m \big[ p(k - am) - p(k - am + 1) \big].$$

## A.2. Pseudocode for CDA

**Algorithm 1:** Pseudocode for CDA. This pseudocode focuses on the salient points of the CDA, namely, how the algorithm draws solutions from the solution space of an underdetermined Diophantine equation to determine the arrangement of hyperstubs around a particular node. Other steps, such as the assignment of hyperedge sizes, are omitted for brevity. The algorithm is described in Section 2.1 and can be viewed in the source code. The output hyperstub degree sequence $H$ must be used as input for a modified configuration model connection process to realise a network, see Section 2.3.

**1 input**: $D = (d_1, d_2, \ldots, d_N)$, $G = \{G_1, G_2, \ldots, G_r\}$

**2 output**: $H \in \mathbb{N}^N_0$

**3 Variables**

**4** $D$: hyperdegree sequence of nodes.

**5** $G$: set of subgraphs; $\Gamma$: number of subgraphs.

**6** $S_k$: subgraph adjacency matrix; $X_k$: solution space for degree $k$.

**7** $\theta$: hyperstub degree sequence.

**8** $P$: hyperstub degree sequence.

**9 for** each subgraph $G$ **do**

**10** % Identify the degree sequences of the subgraphs.

**11** $i_s = \sum_j r_j$

**12** $i_n = $ unique elements.

**13** $i_s = \text{unique}(i_n)$

**14 end**

**15** % Concatenate into a single vector.

**16** $h = (h_1, \ldots, h_r)$

**17** $h_k = X_k(h)$

**18** % $X_k(h)$ denotes a hyperstub arrangement for a degree 4 node.

**19** $X_k = \text{dcwcw}(S_k)$

**20 end**

**21 for** $n = 1, 2, \ldots, N$ **do**

**22** % Take random element from the solution space.

**23** $r = \text{rand}$; $h_n = X_{d_n}(r, \cdot)$

**24 end**

**25** % Concatenate into a single matrix.

**26** $H = (h_1, \ldots, h_N)$

**27 return**


## A.3 Pseudocode for CMA

**Algorithm 2:** Pseudocode for CMA. Other steps, such as ensuring the handshake lemma is satisfied for both lines and subgraphs, are identical to what is used for the UDA and are detailed in Section 2.1 and can be viewed in the Matlab source code. The output hyperstub degree sequence $H$ must be used as input for a modified configuration model procedure to realize a network; see Algorithm 3.

```
1  input : D = (d_1, d_2, ..., d_N), G = {G_1, G_2, ..., G_L}, Y = {S_1, S_2, ..., S_L}
2  output: H ∈ N^{G_N}
3  Variables
4  // D: degree sequence, N: number of nodes,
5  // G: set of subgraphs, L: number of subgraphs,
6  // S: subgraph sequence, g: subgraph adjacency matrix,
7  // y: multinomial proportions, h: hyperstubs in a subgraph, H: hyperstub degree sequence
8  Procedure
9  for Each subgraph, G_i do
10    % Identify the degree sequence, s, of the subgraph.
11    s_i = Σ_j g_{i,j}, s = unique(s), m = length(s)
12    % y reflects the proportions of hyperstubs
13    P_i = (P_{i,1}, ..., P_{i,m})
14    y_i = P_i/||P_i||
15    % The subgraph is decomposed into a hyperstub
16    % y corresponds to the multinomial distribution, M.
17    % so that H_i ∈ N^{s_{i,1} × ... × s_{i,m}}
18    H(j) = M(S(j), y_i)
19 end
20    H_i^' is a sequence of the true stub count
21    H_i^' = H_i · s_i
22    % Sum so that H_i^' ∈ N^{G_N}
23    H(j) = Σ_{i=1}^{L} H(i, j)
24 end
25 while elements of each H_i are non-zero do
26    % Find the largest subgraph degree.
27    [i, j] = Find the largest hyperstub degree.
28    % i.e., the i^th element of H_j
29    % Find all elements of the degree sequence at least this large and
30    % select an element from d at random
31    d' = {d ∈ D : d ≥ s_{i,j}}, m = d (random)
32    % Find E(j) to e' and update
33    % c's available stubs of index j (random)
34    λ = l = H(j), H(j) = 0
35 end
```



## References

1. Girvan, M. & Newman, M. E. J. (2002) Community structure in social and biological networks. *Proc. Nat. Acad. Sci.* **99**, 7821–7826.
2. Bullmore, E. & Sporns, O. (2009) The structure and function of complex networks. *SIAM Rev.* **45**, 167–256.
3. Hopfield, J. J. (1982) Neural networks and physical systems with emergent collective computational abilities. *Proc. Nat. Acad. Sci.* **79**, 2554–2558.
4. Kitsak, S., Gallos, L. K., Hu, H., Klauss, F., Zavorski, Z., Primakul, S., Havlin, S., & Motter, A. E. (2014) Links that speak: the global language network and its association with global culture. *Proc. Nat. Acad. Sci.* **111**, E4913–E4922.
5. Helbing, D. (2013) Globally networked risks and how to respond. *Nature* **497**, 51–59.
6. Jiang, B., Zhao, S. & Yin, J. (2008) Self-organized natural roads for predicting traffic flow: a sensitivity study. *J. Stat. Mech.* **2008**, P07008.
7. Newman, M. E. J., Strogatz, S. H. & Watts, D. J. (2001) Random graphs with arbitrary degree distributions and their applications. *Phys. Rev. E* **64**, 026118.
8. Rao, B. (2006) Performance of networks of artificial neurons: the role of clustering. *Phys. Rev. E* **68**, 045102.
9. Vale, E. (2006) Random networks with tunable degree distribution and clustering. *Phys. Rev. E* **70**, 056113.
10. Bansal, S., Khandelwal, S. & Meyers, L. (2009) Exploring biological network structure with clustered distributions. *Phys. Rev. E* **72**, 036113.
11. Rovsue, S., Karampinos, S. & Meyers, L. (2009) Exploring biological network structure with clustered random networks. *BMC Bioinformatics* **16**, 405.
12. Ritchie, M., Berthouze, L., Bird, T. & Kiss, I. Z. (2014) Higher-order structure and epidemic dynamics in clustered networks. *J. Theor. Biol.* **348**, 21–32.
13. Ritchie, M., Berthouze, L. & Kiss, I. Z. (2016) Beyond clustering: mean-field dynamics in networks with arbitrary subgraph composition. *Journal of Mathematical Biology* **72**, 255–281.
14. Gleeson, J. P. & Bhattacharya, L. (2013) Using ancestry-based Go to sample diversity in graphs satisfying constraints. In *Proceedings of the Companion Publication of the 22nd International Conference on Computer Supported Cooperative Work* (CSCW '13 Companion), 1–4.
15. Robinson, A. E. & Albert, R. (1999) Emergence of scaling in random networks. *Science* **286**, 509–512.
16. Del Genio, C. I., Kim, H., Toroczkai, Z. & Bassler, K. E. (2010) Efficient and exact sampling of simple graphs with given arbitrary degree sequences. *PLoS One* **5**, 1–7.
17. Newman, M. E. J. (2002) Assortative mixing in networks. *Phys. Rev. Lett.* **89**, 208701.
18. Newman, M. E. J. & Girvan, M. (2004) Finding and evaluating community structure in networks. Generating of graphs with prescribed degree correlations. *New J. Phys.* **17**, 083052.
19. Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D. & Alon, U. (2002) Network motifs: simple building blocks of complex networks. *Science* **298**, 824–827.
20. Newman, M. E. J. (2003) Properties of highly clustered networks. *Phys. Rev. E* **68**, 026121.
21. Yaveroğlu, N. & Newman, M. E. J. (2010) Random graphs containing arbitrary numbers of dense subgraphs. *Int. J. Combinatorics* **1**, 311–338.
22. Miller, J. C. (2009) Percolation and epidemics in random clustered networks. *Phys. Rev. E* **80**, 020901.
23. Newman, M. E. J. (2009) Random graphs with clustering. *Phys. Rev. Lett.* **103**, 058701.
24. Karrer, B. & Newman, M. E. J. (2010) Random graphs containing arbitrary numbers of subgraphs. *Phys. Rev. E* **82**, 066118.
25. Klein-Hennig, H. & Hartmann, A. K. (2012) Bias in generation of random graphs. *Phys. Rev. E* **85**, 026101.
26. Irizarry, P. & Ramos, S. M. (2010) The impact of clustering of susceptible contact networks. *PLoS Comput. Biol.* **6**, e1000721 e1000721.
27. Miller, J. & Kiss, I. Z. (2013) Insights from using modern approximations to infections on networks. *J. Roy. Soc. Interface* **9**, 67–73.


**29.** Green, D. M. & Kiss, I. Z. (2010) Large-scale properties of clustered networks: implications for disease dynamics. *J. Biol. Dynamics*, **4**, 431–445.

**30.** Miller, J. C. (2015) Complex contagions and hybrid phase transitions. *J. Complex Netw.*, **4**, 201–223.

**31.** O'Sullivan, D. J. P., O'Keeffe, G. J., Fennell, P. G. & Gleeson, J. P. (2015) Mathematical modelling of complex contagion on clustered networks. *Front. Phys.*, **3**, 71.

**32.** Newman, M. (2010) *Networks: An Introduction*. New York: Oxford University Press, 2010.

**33.** House, T. (2010) Generalised network clustering and its dynamical implications. *Adv. Complex Sys.*, **13**, 281–291.

**34.** House, T., Davies, G., Danon, L. & Keeling, M. J. (2009) A motif-based approach to network epidemics. *Bull. Math. Biol.*, **71**, 1693–1706.

**35.** Xulvi-Brunet, R. & Sokolov, I. M. (2004) Reshuffling scale-free networks: From random to assortative. *Phys. Rev. E*, **70**, 066102.

**36.** Keeling, M. J. (1999) The effects of local spatial structure on epidemiological invasions. *Proc. Roy. Soc. London B*, **266**, 859–867.

**37.** Kiss, I. Z., Green, D. M. & Kao, R. R. (2008) The effect of network mixing patterns on epidemic dynamics and the efficacy of disease contact tracing. *J. Roy. Soc. Interface*, **5**, 791–799.

**38.** Miller, J. C. (2015) Personal communication.
