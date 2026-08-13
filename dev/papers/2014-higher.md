# Higher-order structure and epidemic dynamics in clustered networks

Martin Ritchie <sup>a</sup>, Luc Berthouze <sup>b,c</sup>, Thomas House <sup>d</sup>, Istvan Z. Kiss <sup>a,\*</sup>

<sup>a</sup> *School of Mathematical and Physical Sciences, Department of Mathematics, University of Sussex, Falmer, Brighton BN1 9QH, UK*
<sup>b</sup> *Centre for Computational Neuroscience and Robotics, University of Sussex, Falmer, Brighton BN1 9QH, UK*
<sup>c</sup> *Institute of Child Health, University College London, London WC1N 1EF, UK*
<sup>d</sup> *School of Mathematics, University of Manchester, Manchester M13 9PL, UK*

---

## H I G H L I G H T S

- Networks of equal clustering may show significantly different higher-order structures.
- We present an efficient model counting algorithm.
- This enables finer resolution closures that give more accurate network descriptions.
- We conjecture the SIR model clustering artificially for use in transition states.

---

## A R T I C L E  I N F O

**Article history:**
Received 27 October 2013
Received in revised form
20 December 2013
Accepted 22 January 2014
Available online 30 January 2014

**Keywords:**
Networks
Clustering
SIR modelling
Motifs
Networks
Epidemics

---

## A B S T R A C T

Clustering is typically measured by the ratio of triangles to all higher regardless of whether open or closed. Generating clustered networks, and how clustering affects dynamics on networks, is reasonably well understood but rarely goes beyond considering triangles (e.g., [Eames and Keeling, 2002](reference), [House and Keeling, 2010](reference)), e.g., networks which, despite having the same degree distribution and equal clustering, exhibit different higher-order structures (e.g., different distribution of triangles between nodes, or based on a network composed of four node motifs). To distinguish and quantify these additional structural features, we develop a new efficient algorithm to enumerate and partition open and closed higher-order structures in networks. This provides a more nuanced characterisation of network structure. The impact of such structural differences is then considered: a modified configuration model and two rewiring algorithms. By generating heterogeneous networks with equal clustering but different higher-order structure we show that differences in higher-order structure have a significant effect on epidemic threshold and dynamics. In particular, we observe how differences in higher-order structure impact on epidemic threshold, final epidemic or prevalence levels and time evolution of epidemics. Our results suggest that characterising and measuring higher-order network structures, and exploiting these characterise the better, more accurate models of dynamics on networks.

© 2014 The Authors. Published by Elsevier Ltd. See article note [CC license](reference).


sometimes may be regarded as a by-product of generating low-order structure that can preclude a correct interpretation of the impact of clustering. The paper is structured as follows. We first introduce and describe a set of clustered network generating algorithms. We follow with a presentation of the network metrics (including a description of the motif identifying/counting algorithm) that we propose to quantify similarities and differences between the generated networks. We then analyse and discuss the impact of higher-order structural differences, at identical degree distribution and equal clustering, on SIS and SIR epidemics. Finally, we discuss how our motif-counting results and newly proposed measure for higher-order structures could be used to parameterise pairwise-like models with closure at the level of quadruples.

## 2. Material and methods

### 2.1. Network construction

A significant part of network research relies on networks with arbitrary degree distributions built using the configuration model. This algorithm generates networks where nodes mix at random and where the probability that two nodes are connected is simply proportional to the product of their degree. Such networks coupled with stochastic node dynamics such as SIS, SIR or neural dynamics are amenable to developing macroscopic low-dimensional ODE models that are in excellent agreement with values obtained from stochastic simulations of master-equation models and this displays in the limit of large network size. Whilst such networks can be considered in many cases as realistic or plausible models of some real-world networks, there are many instances where networks have a high clustering structure that typically involves clusters of well connected nodes. Classic examples come from household models used in epidemiology (Ball and Lyne, 2001), and networks of local interactions in ecology. Motivated by this there are a series of theoretical or synthetic network models that can be tuned to display identical levels of clustering (Volz et al., 2011; Karen and Newman, 2010; Newman, 2009; Read and Keeling, 2003; Eames, 2008; Bansal et al., 2009). However, clustering denotes the ratio of closed loops of length three with respect to all possible open triples, irrespective of whether they are closed or not.

The classic algorithms to generate networks with tunable clustering include (a) the spatial algorithm proposed by Read and Keeling (2003) or an iterative rewired proposed by Miller (2009a), (b) configuration model that includes clustering (Karen and Newman, 2010) and (d) the Big-V rewiring algorithm (Newman, 2009; Volz et al., 2011; and Keeling). In a recent study, Green and Kiss (2009) showed that even under identical degree distributions and equal overall clustering, networks built based on different algorithms can display a markedly different 'higher-order structure'. Whilst their analysis involved large scale numerical simulations amongst networks with identical degree distribution and clustering, it did not consider precisely the concept of clustering motifs, i.e. closed loops or higher order structures with four or more nodes. The concept of motif is not

new (Syvens et al., 2005; Karev and Newman, 2010; Volz et al., 2011; Keeling; Tilko; House and Keeling, 2011) and understanding network structure through higher-order motifs is going to provide a level of detail which cannot be articulated by open or connected triples alone. Below we provide a brief description of the clustered network construction algorithms used in this paper.

#### 2.1.1. Big-V rewiring

The 'Big-V' is an iterative rewiring algorithm that can introduce clustering into any given network and is commonly used by network scientists (Bansal et al., 2009; House and Keeling; Green and Kiss, 2009). At each iteration step, a chain of 5 nodes/edges $(a-i-x-j-y)$ is selected at random and a clone network is generated where the links $(a-i)$ and $(x-y)$ are broken and the edges $(a-y)$ and $(i-x)$ are created. This leads to a single chain of 5 nodes being broken into a triangle and a disconnected pair, see Fig. 1. Local clustering for each node is calculated as well as all of the sub-network properties for both the original and cloned networks and the new configuration is kept only if the level of clustering has increased.

#### 2.1.2. MD motif/composition rewiring

MD (Motif Driven rewiring) is an iterative rewiring algorithm that starts with a collection of complete sub-networks that are disconnected from one another and rewires edges randomly to reduce the clustering from its maximal value of 1 to the desired level. The following steps are performed:

i. Initialise a network that is composed of $m$ complete motifs each with $n$ members so that $N = nm$ and $\langle k \rangle = n - 1$.
ii. Set the desired level of clustering.
iii. For the first step only, select at random two local edges, cut them, and swap the stubs to form new edges. Mark the pair of new edges as global.
iv. Select a local and a global edge, cut them, and swap the stubs to form new edges. Mark the pair of new edges as global.
v. Check the global clustering, if the desired level has not been reached, repeat step (iv).

Fig. 2 illustrates this process being performed on a complete motif with 4 members. It should be noted that this method may work with a heterogeneous degree distribution in which case the network would need to be initialised with motifs of $k+1$ nodes for each different degree $k$. MD has the significant advantage that it is computationally cheap to implement and during construction, network properties can be calculated analytically (see Appendix A1).

#### 2.1.3. CCM (Clustered Configuration Model)

It is possible to modify the configuration model (Miller, 2009; Volz, 2011) so that it constructs networks using specified motifs. Karen and Newman (2010) and Volz et al. (2011) have shown how to build networks based on specified proportions of open and triangle motifs. This idea may be easily extended to allow for larger and more exotic

![A Big-V rewiring diagram showing: (a) a chain of 5 nodes with edges, (b) the result after rewiring where a triangle is formed and a disconnected pair remains, and (c) an independent outcome where the algorithm proceeds to find a new chain.](figure)

**Fig. 1.** A single Big-V rewiring: (a) identify a chain of 5 nodes with 4 edges and (b) if edges $(a-i)$ or $(x-y)$ are already part of a triangle the cuts will not be made, otherwise rewiring is performed, and (c) independent of the outcome of (b) the algorithm will proceed to find a new chain.


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


reverse has not been counted as an open path then count an open path of length $l + 1$.

(d) For all closed paths of length $l + 1$ remove circular and reverse circular permutations.

(e) Categorize each closed path by its completeness, i.e., the number (if any) of diagonals in a square.

## 2.1 Dynamics on networks

To establish the overall impact of higher-order network structure, simulations of various dynamics are performed on the generated networks. First, we use the Markovian SIR (susceptible-infectedrecovered) model with a per-contact infection rate $\tau$ and recovery rate $\gamma$. All simulations are performed using the Gillespie (1976) algorithm. To assess the impact of loops and cycles, we also simulate the SIS epidemic which is more likely to highlight differences in the cycle/motif composition. We shall see that structural differences between networks lead to differences in distribution and clustering manifest in epidemiological differences with regard to dynamics on the networks. Previous work (Rand, 1999) used Kirkwood's superposition approximation to predict the effect of order-four structure on epidemic dynamics. In particular, equations for a GCC consisting of $N$ nodes connected in a single GCC were considered and their corresponding system of triple-wise ODs was derived. The equations corresponding to order-three motifs were left open, however, equations that correspond to order-four motifs were closed using order-three terms such that it was only in the limit $N \to \infty$ that equations became exact. For SIS dynamics it was conjectured that the presence of empty square structures reduces the endemic state for all levels of $\phi$ (varying only $\phi$ and the prevalence of empty squares whilst keeping all other parameters unchanged), and complete squares may increase or decrease the endemic state and that diagonal squares had very little effect on the endemic state. In the following we make comparisons between networks that have varying distributions of order-four motifs. We expect that networks with markedly different order-four motif distributions produce different epidemiological behaviour.

## 3. Results

Using the various construction algorithms, we give an overarching analysis of structural features of the networks built with the same degree distribution and same levels of classic clustering. All networks used are homogeneous with $\bar{k} = 4$ and have carefully controlled structures/loops whilst keeping the complexity to a manageable level. We carry out this analysis for a number of clustering values (i.e. $\phi = 0.2, 0.4, 0.8$) to measure and evaluate the extent to which clustering can emerge from, or determine, different configurations of order-four structures.

*3. Overall feature and structure of the network*: Graph (Barabasi et al., 2000) network visualisations of networks being constructed by the proposed algorithms, see Fig. 4. In these figures nodes are colour coded according to clustering, with more loosely or un-clustered nodes coloured with nuances closer to the red end of the spectrum, and more highly clustered nodes coloured with shades closer to the blue end of the spectrum. The figure clearly illustrates that the CCM algorithm gives rise to networks with an initially higher clustering value, whilst the rewiring algorithms (i.e. Big-V and MD) construct networks with more uniformly and highly clustered nodes (Fig. 4). It is also evident that this difference translates into a more modular structure for the rewired networks. The CCM

networks stand out as being structurally different from the networks generated by the other algorithms, as well as being homogeneous in degree, they are also homogeneous in structure.

*2. Distribution of clustering and centrality*: The almost homogeneous distribution of the local clustering (see Fig. 5) and betweenness centrality (see Fig. 6) of the CCM networks is expected since by construction every node has the same local structure, i.e. $\phi = 0.8$ we know that each node has a quintuplet of stubs with probability $p_1$ or a complete square and a triangle with probability $p_2 = 1 - p_1$. When $\phi = 0.4$ every node is a member of one triangle and one complete square.

The Big-V algorithm introduces clustering in a more heterogeneous way, generating one of the highest local clustering values in the range $0 \leq \phi \leq 0.5$. The MD algorithm provides the largest degree of clustering with half of the nodes having clustering in the range $0.2 \leq \phi \leq 0.6$. The box plot (Fig. 5) of local clustering shows the properties of the MD network tend to be broadly unchanged. These complete motifs must be compensated with other parts of the network being decomposed into a much more random graph-type structure. The MD algorithm relies on random edge swapping to decompose the network into a more random structure while preserving the local clustering. A few leftover fully connected motifs which are only destroyed at the same levels of clustering $\phi$ decrease. Even if the number of such motifs is still large but the overall, desired, clustering is moderate, the connected parts of the network have to be left less structured.

The plot of betweenness centrality (Fig. 6) illustrates this point more explicitly. Nodes in a random graph-type network will have a low betweenness centrality whilst those that act as bridges between the random and the rest of the network will be more highly ranked. We observe that the betweenness centrality distribution is much more dispersed for MD networks. The removal of nodes (with a high betweenness centrality rank) is more likely to have a bigger impact on dynamics residing in the network when built with the MD algorithm.

*3. Connected component analysis*: For a well connected network with low clustering, by definition with $\phi \to 0$ and keeping the structure of the networks unchanged (see Fig. 7) i.e. the entire network is contained within a single GCC, we can see that all clustering are resilient to the removal of a relatively small number of edges. For example, the GCC still spans a large proportion of the network even if 50% of the edges are removed. This behaviour has been previously noticed (Gleeson et al., 2010).

The size of the GCC decreases with increasing clustering across all different values of $\phi$ (from top row Fig. 7 to the bottom row Fig. 7). This behaviour is expected since in any network where $\bar{k} \leq \bar{k}_c$ higher clustering is only achievable through more complexity/reducing long cycles. Given the initialisation of MD networks where $\phi \to 1$ and the network is composed of disconnected cliques, this decrease is further reflected in the step behaviour of the plots when $\phi = 0.8$ (bottom row Fig. 7). Despite decreasing in size of the GCC, Fig. 4 shows that MD networks are the most robust to edges being removed. We stress that this is not about global edge removal but rather if a particular edge is structurally more important than another. MD networks are extremely sensitive to edges being removed: we observe a dramatic decrease in size and a strong dependency on components of size 6 or fewer. The MD algorithm fails to protect a highly connected network in as far as to leave some motifs weakly connected to the GCC at this level of clustering. The Big-V algorithm produces networks with a relatively well connected GCC but still exhibits a mild sensitivity to the removal of edges.


![Four example networks showing heterogeneous networks with all parameters held equal, N=400, alpha=1, and rho=0.4. The networks at the low end of the spectrum have local clustering, and those at the high end have high local clustering. The interpretation of the references to colour in the figure caption, the reader is referred to the web version of this paper.](figure)

**Fig. 4.** Example networks. Heterogeneous networks with all parameters held equal: $N = 400$, $\alpha = 1$, and $\rho = 0.4$. The networks at the low end of the spectrum have local clustering, and those at the high end of the spectrum have high local clustering. The interpretation of the references to colour in this figure caption, the reader is referred to the web version of this paper.

![Box plots showing local clustering. Results of local clustering measured from 20 heterogeneous networks, N=100, alpha=1. The size of the network ego chosen in its clustering is based on the number of nodes observed in the plot, and the inter-clustering between neighbours of a given node provides the largest increase in local clustering.](figure)

**Fig. 5.** Local clustering. Boxplots of locally observed fully connected subgraph clustering from 20 heterogeneous networks, $N = 100$, $\alpha = 1$. The size of the network ego chosen in its clustering is based on the number of nodes observed in the plot, and the inter-clustering between neighbours of a given node provides the largest increase in local clustering.

![Distribution of heterogeneous clustering. Box plots of motif betweenness centrality measured from 20 heterogeneous networks, N=100, alpha=1. Betweenness centrality is apparent at moderate levels of clustering rho=0.4, 0.2. The CM and Erd\H{o}s-Renyi clustering algorithms give structurally similar distributions. A large spread of heterogeneous clustering values is a sign of a high spread of outlier nodes.](figure)

**Fig. 6.** Distribution of heterogeneous clustering. Box-plots of motif betweenness centrality measured from 20 heterogeneous networks, $N = 100$, $\alpha = 1$. Betweenness centrality is apparent at moderate levels of clustering ($\rho = 0.4, 0.2$). The CM and Erdős–Rényi clustering algorithms give structurally similar distributions. A large spread of heterogeneous clustering values is a sign of a high spread of outlier nodes. The distribution of a high number of outliers observed in each network provides the largest increase in global clustering (providing the longest path across a network).

4. *Motif statistics for all network types* (Table 2) shows that third-order clustering convey little information about order-four-but-not-three structures. As we might expect, the number of complete structures of 4-nodes increases with clustering, and for high levels of clustering and for high levels there is a deficit of complete squares. The algorithms' lack of control of order-four structure is apparent at moderate levels of clustering ($\rho = 0.4, 0.2$). Reading column-wise down each column shows particular third-order structures across networks of equal clustering. The difference in $\rho_e$ is due to triangles which do not map to triangles. We note that the triangles are not measured by this metric. The distribution of triangles is important at higher levels of clustering

where they often share edges as overlap to form order-four structures.

Reading column-wise down the columns we see a more particular specialisation of squares with clustering: at clustering such as clustering such as clustering such as clustering, there is a deficit of complete squares with increased clustering among complete squares we see a general trend of increasing complete square prevalence with increased clustering. Nodes may have a count of ten complete squares associated with them when they are members of a complete


![Six panels showing edge percolation plots as a function of transmissibility p, comparing CGM and Big-V network models with SIR epidemic dynamics. Top row shows three plots for CGM; bottom row shows three plots for Big-V. Each plot contains curves for different clustering levels (A=0.05, A=0.118, A=0.168) with lines for final size, giant component, and simulations.](figure)

**Fig. 7.** Edge percolation plots. Frequency of component sizes as edges are removed from the network with probability $p$. Results are taken from homogeneous networks with $\langle k \rangle = 6$ and $n = 1000$. CGM networks are shown at $p=0.4, 0.6, 0.8$ and Big-V networks at $p=0.3, 0.5$ respectively. Comparing percolation clustering at $n = 0.05$, $n = 0.118$ and $n = 0.168$ respectively. Each line represents a different value of $n$, varying from 0.05 (blue) to 0.168 (red). Note the difference in the $x$-axis between the two panels; the sudden jumps are a result of a strong dependency on a small of certain size in the dynamics of the network. $N_{cl}$ is the dimension of the network, $N_{cl} = 3 \times$ (number of triangles in network)/(number of connected pairs of edges in network).

and isolated, no node structure. At all levels of clustering the probability of finding an empty square associated with a node or a triangle is identical. This means that empty squares do not contribute to clustering.

Finally, Fig. 7 also reveals that networks generated by the Big-V algorithm contain empty square motifs with very low frequency. The algorithm searches for unshared triples (sequences of three nodes connected by two distinct edges) and motifs that may be constructed out of triangles can be expected to Big-V networks in any significant quantity. The MD algorithm also generates few empty square motifs and the CGM algorithm will only include them by specification.

## 3.1. *Dynamics on the networks: evaluation and comparison*

To investigate the effect of network structure on disease we use the two classic (e.g. SIR and SIS) epidemic dynamics on networks. Starting with the simplest situation, and an epidemic: for triangles and for empty squares, the dynamic is the same in both cases; second, the two infected nodes then compete for the same remaining susceptible. For empty squares, there is a race to infect a similar but less dramatic. Fig. 7 shows that the initial epidemic spread is slower for networks which have loops. By opening a closed motif while preserving degree, two new individuals must be added to the other of competition is inversely proportional to the degree, while in empty squares it increases with a higher degree, for a given level of competition.

When simulating epidemics on networks with $p = 0.2$, the CGM networks show a slower spread of infection (Fig. 5). At this level of clustering the CGM algorithm breaks a quintuplet of stubs into all possible triangles and chains, and the epidemic can be expected with $p = 1/2$. Thus, the CGM networks exhibit areas of high clustering in which the disease will spread more slowly than in areas of low clustering. At $p = 0.2$ the CGM networks exhibit a slower spread of infection (Fig. 5). At higher clustering levels, any one susceptible has a greater likelihood of having already been infected. In Fig. 5 we see the difference between $\Delta = 0.2$ and $n = 0.4$ with a less dramatic spread through triangles. Strong clustering creates an even higher level leads to the network breaking down into many small disconnected components, which impedes transmission of the epidemic. This effect is expected to apply in networks where many chains of nodes are cut by limited or no connectivity between the highly connected clusters.

Clustering can have the effect of creating sub-networks that contain clustered motifs that are poorly connected to the rest of the network. This will impede transmission on its own, but may result in life-type processes, which when recovered significantly hinder the propagation of the epidemic. Both of the existing algorithms


![Bar charts showing pair-state distribution of the number of connected pairs of colour-four motifs for all presented network models. See Appendix A.1 which details how motifs are counted.](figure)

**Table 2**

For each level of clustering the table has been sorted in ascending order (thus ascending number gives the most pairs are of the desired colour pair). The value represents the proportion of all closed quadruples that are colour-four compared using unique counts.

|         | $\phi(2)$ | $\phi(3)$ | $\phi(4)$ | $\phi$ |
|---------|-----------|-----------|-----------|--------|
| COK-0.2 | 0.0093    | 0.0087    | 0.0080    | 0.0086 |
| RG-0.2  | 0.0117    | 0.0095    | 0.0080    | 0.0098 |
| MG-0.2  | 0.0210    | 0.0087    | 0.0080    | 0.0116 |
| COK-0.4 | 0.0248    | 0.0187    | 0.0044    | 0.0159 |
| RG-0.4  | 0.0289    | 0.0190    | 0.0044    | 0.0175 |
| MG-0.4  | 0.0372    | 0.0183    | 0.0044    | 0.0216 |
| RG-0.8  | 0.5518    | 0.0033    | 0.0062    | 0.2244 |
| MG-0.8  | 0.5695    | 0.0044    | 0.0062    | 0.2299 |

produce nodes with high betweenness centrality when compared to the LCM algorithm. It has previously been noted that the MG networks are particularly dependent on isolated motifs (this is especially prominent with higher clustering). This influences the SIR dynamics at moderate levels of clustering. The CCM and Rg V networks have the more consistent connectivity throughout the network and, consequently, the more consistent dynamics.

Using such simple dynamics with few states and simple transition probabilities, the subtlety in the differences is somewhat expected. Nevertheless we foresee that other more complex dynamics, such as intrinsical dynamics or modified voter model, where transitions may

depend on membership within certain motifs and transitions do not always scale linearly with the state of the neighbouring nodes (e.g. as in [illegible]) may be interesting contexts that lead to more marked differences.

## 4. Discussion

The development of models that capture epidemic or other dynamics on networks is guided, to a great extent, by the structure of the network and how that structure can be used to understand the impact of degree distribution, or heterogeneity in contact. This was closely followed by results capturing preferential mixing, where nodes of similar degrees can be either more likely (assortative mixing) or less likely (disassortative mixing) to be connected. Despite the fact that most real world networks show some clustering, at least accounting for their effect, but looked at clustering, the tendency for neighbouring nodes also to be mutually connected or to share of each other. In the area, progress is still being made and the picture may now complete by any means.

We have proposed a novel and flexible graph construction method designed to be able to control and tune properties such as degree distribution, mixing, clustering and so forth. However, as shown in this paper, more subtle network properties (i.e., beyond one- and two-neighbourhood properties) can and will have an effect on higher-order structure and this can be significant and cannot be disregarded. For example, at high values of clustering generated based on the spatial algorithm (Neal and Kenlay, 2012), the networks


![Two panels showing random vs epidemic comparison plots for 20 homogeneous networks with N=1000. Left panel shows delta=2, right panel shows delta=3. Each panel contains time series curves for I, L1, and L2 variables, with simulation results (colored lines) compared to theoretical predictions (dashed lines).](figure)

**Fig. 9.** Random vs supra comparison. 20 homogeneous networks were generated with $N=1000$, $\delta=2.5$ and $\rho=0.6598$. The plots correspond to averaging 100 epidemic simulations on each of the networks with parameters $\tau=0.1$ and $\delta$ suitably chosen. The networks labelled 'supra' were constructed by altering their degrees to be all equal to $\delta$ and $\rho$, respectively.

![Two panels showing SIR and SIS dynamics for 20 homogeneous networks. Each panel shows time series for I, L1, and L2 variables comparing simulation (colored lines) with theoretical predictions (dashed lines).](figure)

**Fig. 10.** SIR and SIS dynamics. 20 homogeneous networks were generated with $N=3000$ and $\delta=5$ and the results show the average of 100 100-epidemic epidemics on each network realisation. The epidemics were run with parameters $\tau=0.1$ and were seeded with 5 infectious nodes. The top and bottom rows show the prevalence levels for a SIR and SIS epidemic, respectively.

become more asymmetrically mixed. Such effects are to be expected since the network is a coherent structure which reacts to each perturbation, such as rewiring or other means of losing properties. In general, we expect that Rig-9 rewiring will be the most random way to introduce clustering without model-specific artefacts. However, this comes at what is straining to prodigious computational resources. For this reason, MI is much more analytically tractable, but is a long way from randomly introduced clustering. In this context, MI can be viewed as a computationally tractable way of introducing higher-order structure that is similar to Rig-9 that produces very similar network phenomenology.

In this study, we highlight how using two algorithms that generate networks with tunable clustering do lead to different higher-order structures, such that networks with the same degree distribution and level of clustering can yield different dynamics on the networks. In order to evaluate differences in higher-order structures we have extended the concept of clustering and proposed some measures to evaluate and quantify the frequency of structures composed of four nodes.

The measures we have proposed are ratios of the uniquely rooted, directed, closed paths that emerge as a consequence of the connected structures of four nodes. This is conceptually convenient but these values may not be suitable for use in low-dimensional ODE approximations. For this reason, in Appendix A.1 we describe a method of using purely unique counts (see Appendix A.1) and patch a different value when the unique counts are used. In Appendix A.2 we hypothesise the correct counts of motifs and paths for use in clustering-type ratios. Whilst counting uniquely significantly reduces


![Plots of final endemic size and endemic equilibrium fractions versus the rate of r increasing from r=0.1 to r=1.0 in increments of 0.5. Dotted networks were generated via Gillespie simulations performed for each value of r. The networks were homogeneous with N=5 and k=1000.](figure)

computational complexity, it has the slight disadvantage that it does not provide the multiplying type of counting used in pairwise models. In the Appendix, we conjecture that this can be easily overcome by simply multiplying by *k*, and this can be confirmed numerically by the simulations performed here.

It has been demonstrated that care needs to be taken when trying to extend modelling to clustered networks. Whilst models for single clustered networks composed of exclusively non-overlapping triangles and edges have been developed, it is yet to be more meaningfully extended to topologies with a higher order of clustering. Networks such as a square with a diagonal or a fully connected square may fulfil some function depending on the area of application (e.g. genetic networks), and therefore developing methods that are capable of quantifying this correctly is crucial for further model development.

Many extensions for this work exist ranging from considerations around higher-order structure, algorithmic efficiency in measuring these and developing stochastic network models that allow clearer interpretation control of not only lower, but also higher-order structures.

## Acknowledgements

Martin Ritchie acknowledges funding for his PhD studies from EPSRC (Engineering and Physical Sciences Research Council) and the University of Sussex. Thomas House acknowledges funding from MRC and would like to thank Charo I. del Genio for discussions on the Motif Decomposition algorithm.

## Appendix A

### *A.1. Motif decomposition analysis*

It is possible to write down the dynamics for the SIS process (Section 6.1.1) in the limit of large networks by decomposing motifs into hyper motifs (considering each motif at a higher level of scale) and considering the links between them. We now consider the process being performed in a homogeneous network with *N* nodes and each node having exactly degree *k* and using the notation that $S_k = \langle S \rangle$ and $I_k = \langle I \rangle$. In this scenario, it is possible to write equations for the normalised count of each hypermotif:

$$\dot{S}_k = -\tau S_k I_k + \gamma I_k, \tag{13}$$

$$\dot{S}_{kk} = -\hat{\tau}(S_{kk}I_k + S_{kk}S_kI_{kk}/S_k^2) + \hat{\gamma}(S_{kI_k} - S_{kk}), \tag{14}$$

$$\dot{S}_{kI_k} = -\hat{\tau}(S_{kk}I_k + S_{kI_k}I_k - S_{kk}S_kI_{kk}/S_k^2 - S_{kI_k}^2/S_k) + \hat{\gamma}(I_{kI_k} - S_{kI_k} - S_{kI_k}), \tag{15}$$

$$\dot{I}_{kk} = -\hat{\tau}(I_{kk}I_k + I_{kk}I_k) + \hat{\tau}S_{kI_k} - \hat{\gamma}(I_{kk} + I_{kk}), \tag{16}$$

$$\dot{S}_{kk^2} = -\hat{\tau}(S_{kk^2}I_k + 2S_{kk^2}S_kI_{kk}/S_k^2) + \hat{\gamma}(2S_{kI_k^2} - S_{kk^2}). \tag{17}$$

These equations can be solved for initial conditions $S_k(0) = 0.9$ and the remaining fraction $I_k(0) = 0.1$ with re-noting a dot is just included for clarity and can be set to 1 with no loss of generality. The process stops at a time *t*\* when it is determined that clustering has been achieved:

$$T_c^1(k) = \frac{\phi}{k}, \tag{18}$$

where $T_c$ denotes the number of triangles associated with each node in the network. These equations can be solved for the motif structure and inserted into Eqs. (1)–(5) to obtain a prediction for motif structure. The hyper-motifs can be done for a star but quickly become non-intuitive. It is also possible to use the quantities in Table 1 to derive epidemic final size and other attributes. Reading


Fig. 2 along side Eqs. (15)–(17) it should be noted that for hypernode $Q_4$ there is only one possible way it can decompose to $Q_4$, by removing edges according to the diagonal. Considering Eq. (17), any one of the $Q_3$'s four square edges may be deleted resulting in a single $Q_3$ and $Q_4$ hyper-node, any one of the $Q_4$'s four edges may be deleted resulting in four $Q_2$ hyper-nodes and any one of the $Q_4$'s three edges may be deleted resulting in three $Q_1$ hyper-nodes (Table 3).

### A.2. Motif counting algorithm

Below we introduce some notation in order to describe correctly and un-ambiguously the counting algorithm.

*Path:* A path *P* is an ordered tuple $(i_1, \ldots, i_n)$:
- $P_e$, the *n*th node of path *P*
- $P^L$, the head operator such that $P^L(P)$ returns the *e* first nodes of path *P*
- $P^T$, the tail operator such that $T^L(P)$ returns the *e* last nodes of the path *P*
- $R$, the reverse operator such that $R(i_1, \ldots, i_n) = (i_n, \ldots, i_1)$
- $C(P)$, the set of circular permutations of path *P*
- $T(P)$, the set of all reverse circular permutations of path *P*: $TR(P)$, $R(P) \cup CR(P)$
- $A$, the adjacency matrix, $A = A^T$ and with $Tr(A) = 0$
- $|.|$, a set
- $\Pi$, the set of non-trivial paths of length 1 (3 = number of edges).

**Algorithm 1.** Pseudo code for the motif counting algorithm.

> *In the following process is applied iteratively to determine non-trivial paths (open paths or closed paths) of length $l + 1$ given non-trivial (non loops) paths of length l. The description below is not specific to a single length but assumes $l \leq 3$. Then The uniquely counted set of paths of length n: $C_n = \{p_1, p_2, \ldots, p_k\}$ where $p_i \in \Pi$ and $i \neq j$.*
>
> *initialisation:*
>
> $C_l = \emptyset$, the closest paths of length $l = 3$
>
> $\Pi_{l-1} = \emptyset$, in Open paths of length $l = 3$
>
> *for All paths P in $\Pi_l$ do*
>
> &nbsp;&nbsp;&nbsp;&nbsp;**for** All nodes $n_i \in V \, A_{P_e n_i} = 1 \, \& \, n_i \notin P$ **do**
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$nP \leftarrow (P, n_i)$ // new path
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $n_i = nP^L_i$ **&** $nP \in C_l$, **&** $R(nP) \notin C_l$ **then**
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\mathcal{O}_{l-1} \leftarrow nP$
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**if** $n_i \neq nP^L_i$ **&** $nP \in \Pi_l$ **&** $R(nP) \notin \Pi_l$, **then**
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Pi_{l-1} \leftarrow nP$
>
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;**end**
>
> &nbsp;&nbsp;&nbsp;&nbsp;**end**
>
> *// These exclude symmetric paths but not circular permutations.*
>
> *end*
>
> if $nP^L \in C(P)\Pi_{l-1}, |, R(P\Pi_{l-1})$ **then**

**Table 3**

Table of hyper nodes induced by *l* with the number of nodes received $n_l$, the number of local links *L*, the number of global stubs $\sigma_g$, and the number of triangles *T* counted in both directions from each hyper-node $H_l$. The denotes that the addition is dependant on the case in question.

| | $Q_1$ | $Q_2$ | $Q_3$ | $Q_4$ |
|---|---|---|---|---|
| $n_l$ | 1 | 1 | 1 | 1 |
| L | 0 | 0 | 1 | 1 |
| $\sigma_g$ | 1 | 2 | 2 | 1 |
| $T_l$ | 0 | 0 | 0 | 0,1,1 |

$|\Pi_{l-1}| \leftarrow (\Pi_{l-1}, P)$,

$|\Pi_{l-1}| \leftarrow (\Pi_{l-1}, P)$

**end**

*// Removes circular permutations.*

**end**

**for** All paths $P \in \mathcal{O}\Pi_{l-1}$ **do**

&nbsp;&nbsp;&nbsp;&nbsp;**if** $P^L \in TR(P)$, **&** $P^T \notin TR(P\Pi_{l-1})$, **then**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Pi_{l-1} \leftarrow P$

&nbsp;&nbsp;&nbsp;&nbsp;**end**

**end**

**for** All paths $P \in C\Pi_{l-1}$ **do**

&nbsp;&nbsp;&nbsp;&nbsp;**if** $P^L \in TR(P)$, **&** $P^T \notin TR(C\Pi_{l-1})$, **then**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;$\Pi_{l-1} \leftarrow P$

&nbsp;&nbsp;&nbsp;&nbsp;**end**

**end**

*// Removes reverse circular permutations.*

### A.3. Motif counting: unique to multiplied permutations

In this paper all order-four clustering type ratios use unique counts. Ratios based on unique counts will give different values to ratios based on multiplicative counts. As an example of multiplicative counting, classic clustering is defined as

$$\phi = \frac{3 \times \text{(triangles)}}{\text{(triples)}} \tag{18}$$

where $(\cdot)$ denotes the number of triangles, and $(\cdot \cdot)$ the number of closed and unclosed length three paths (doubly counted) in the network. If unique counts are used then we have $\phi_{\text{unique}} = \phi/3$.

We have computed the unique order-four counts in order to improve the computational performance of our algorithms. However, when comparing or modelling the unique counts to multiplied counts to correspond to the multiplicative equivalent, correct multiplying factors need to be determined. This appears to be the number of automorphisms associated with each motif type or path length: a triangle has six and a square of length three has two automorphisms.

Let $A = (a_{i,j})$, $i, j = 1, \ldots, N$, be the adjacency matrix of an undirected network with no self loops i.e. $A = A^T$ and $A_{i,i} = 0 \forall$ for any $i = 1, \ldots, N$. It is possible to obtain the multiplicative counts from the adjacency matrix *A*. Summing all entries:

$$|.| = \sum_{i,j} A_{i,j} \tag{19}$$

This counts twice the number of real or uniquely counted edges in the network. It is possible to count more complex paths as well

$$|.| = \sum_{i,j,k} A_{i,j} A_{j,k} - \sum_{i,j} A_{i,j} \tag{20}$$

yielding all connected structures of 3 nodes (closed and unclosed), similarly

$$|.| = \sum_{i,j,k} A_{i,j} A_{j,k} A_{k,i} \tag{21}$$

yielding six times the number of unique triangles. It is also possible to count six different closed triples contained within a single triangle, since each set of three vertices induce permutations clockwise and anticlockwise about the triangle. The by-directional counting is important so that this method is consistent when considering directed networks. Following the same counting methodology it is possible to count order-four structures:

$$|\Box| = \sum_{i,j,k,l} A_{i,j} A_{j,k} A_{k,l} A_{l,i} - 4 \sum_{i,j,k} A_{i,j} A_{j,k} A_{k,i} - \tag{22}$$


**Table 4**
Combinations of $i$, $j$, $k$ and $l$ that satisfy (27).

| $i$ | $j$ | $k$ | $l$ |
|-----|-----|-----|-----|
| 1   | 2   | 3   | 4   |
| 1   | 3   | 2   | 4   |
| 1   | 4   | 2   | 3   |

In this form we see that it is possible to compute the individual counts using the following identities:

$$[\stackrel{\frown}{23}] = \sum_{i,j,k,l} \delta_{ij} \delta_{jk} \delta_{kl} \delta_{li} \delta_{lm} \delta_{mi}$$
(24)

$$[\stackrel{\frown}{23}] = \sum_{i,j,k,l} \delta_{ij} \delta_{jk} \delta_{kl} (1 - \delta_{il})$$
(25)

$$[\stackrel{\frown}{23}] = \sum_{i,j,k,l} \delta_{ij} \delta_{kl} (1 - \delta_{jk}) (1 - \delta_{il})$$
(26)

$\delta_{ij}\delta_{kl} = 1$ is counted 24 times. $[\stackrel{\frown}{23}]$ is counted 4 times and $[\stackrel{\frown}{23}]$ is counted 8 times, equal to the number of automorphisms associated with each motif type. By listing the different combinations of $\{i,j,k,l\}$ that satisfy (for diagonal squares):

$$\delta_{il}\delta_{jk}(1-\delta_{ij})(1-\delta_{ik}) = 1,\tag{27}$$

such that $i \neq j, k \neq l$ it is possible to gain insight into the cardinality of this count. Consider a diagonal square as orientated in the notation $[\stackrel{\frown}{23}]$ labelled starting at the top left node in a clock wise direction: $i$, $j$, $k$ and $l$. By listing the combinations in this way we have not included the trivial rotation symmetry (which goes to itself, the orbit between $j$ and $l$: the orbit between $i$ and $k$) and reflections (which includes both orbits) have been excluded.

Currently, based on an online and numerical tests, we conjecture that this is the only correct way scale-up from motif to multiplex motif counts. This method of counting is thorough but it would not be computationally reasonable size since it has complexity $O(N^4)$ for order-4 structures.

## References

Ball, F., Sirl, D., 2009. Stochastic SIR-type epidemics among a population partitioned into households and workplaces. Adv. Appl. Probab. 41 (1), 73–101.

Bansal, M., Bhosekar, A., Bhavanam, M., Iyengar, V., 2006 (supra). An Open-Source Software for Building and Analyzing Complex Networks. Available: http://cran.r-project.org/web/packages/igraph.

Barabasi, A.-L., Albert, R., 1999. Emergence of scaling in random networks. Science 286 (5439), 509–512.

Brouwers, L., Cakici, B., Camitz, M., Tegnell, A., Boman, M., 2010. Socially distanced infectious disease modeling of the 2009 H1N1 pandemic in Sweden. PLoS ONE 5 (9).

Cauchemez, S., Bhatt, S., et al., 2014. Unraveling the drivers of MERS-CoV transmission. Epidemics 6, 1–5.

Crofts, J.J., Higham, D.J., 2009. A weighted communicability measure applied to complex brain networks. J. R. Soc. Interface 6 (33), 411–414.

Danon, L., Read, J.M., House, T.A., Vernon, M.C., Keeling, M.J., 2012. Social encounter networks: characterising Great Britain. Proc. R. Soc. Lond. Ser. B: Biol. Sci. 280 (1765), 20131037.

Dorogovtsev, S.N., Goltsev, A.V., Mendes, J.F.F., 2008. Critical phenomena in complex networks. Rev. Mod. Phys. 80 (4), 1275.

Eames, K.T., Keeling, M.J., 2002. Modeling dynamic and network heterogeneities in the spread of sexually transmitted diseases. Proc. Natl. Acad. Sci. 99 (20), 13330–13335.

Erdős, P., Rényi, A., 1960. On the evolution of random graphs. Publ. Math. Inst. Hung. Acad. Sci. 5 (1), 17–60.

Estes, C., Abdel-Kareem, M., Bhowmik, D., 2012. Hepatitis C transmission: Epidemiologic considerations. Proceedings of the Royal Society of London, Series B: Biological Sciences. 280 (1765).

House, T., Keeling, M.J., 2011. Insights from unifying modern approximations to infections on networks. J. R. Soc. Interface 8 (54), 67–73.

House, T., Ross, J.V., Sirl, D., 2013. How big is an outbreak likely to be? Methods for epidemic final-size calculation. Proceedings of the Royal Society of London, Series B: Biological Sciences 280 (1765), 20122, 214.

Ilinskaya, O., Litovchenko, V., Otkidach, D., 2014. Clique-based algorithms for network motif discovery in temporal networks. Proceedings of the Royal Society of London, Series B: Biological Sciences.

Janssen, J., Laarraj, A., 2013. Spread of infection in directed and weighted social networks. BMC Syst. Biol. 7 (S), 74.

Jeong, H., Mason, S.P., Barabasi, A.-L., Oltvai, Z.N., 2001. Lethality and centrality in protein networks. Nature 411 (6833), 41–42.

Kemper, J.T., 1980. On the identification of superspreaders for infectious disease. Math. Biosci. 48 (1), 111–127.

Keeling, M.J., 1999. The effects of local spatial structure on epidemiological invasions. Proc. R. Soc. Lond. Ser. B: Biol. Sci. 266 (1421), 859–867.

Keeling, M.J., Eames, K.T.D., 2005. Networks and epidemic models. J. R. Soc. Interface 2 (4), 295–307.

Kermack, W.O., McKendrick, A.G., 1927. A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London, Series A: Mathematical and Physical Sciences 115 (772), 700–721.

Klovdahl, A.S., 1985. Social networks and the spread of infectious diseases: the AIDS example. Soc. Sci. Med. 21 (11), 1203–1216.

Kucharski, A.J., Kwok, K.O., et al., 2014. The contribution of social behaviour to the transmission of influenza A in a human population. PLoS Pathog. 10 (6).

Lindquist, J., Ma, J., Van den Driessche, P., Willeboordse, F.H., 2011. Effective degree network disease models. J. Math. Biol. 62 (2), 143–164.

Lloyd-Smith, J.O., Schreiber, S.J., Kopp, P.E., Getz, W.M., 2005. Superspreading and the effect of individual variation on disease emergence. Nature 438 (7066), 355–359.

Manitz, J., Kneib, T., Schlather, M., Helbing, D., Brockmann, D., 2014. Origin detection during food-borne disease outbreaks — a case study of the 2011 EHEC/HUS outbreak in Germany. PLoS Curr. 6.

Marceau, V., Noël, P.-A., Hébert-Dufresne, L., Allard, A., Dubé, L.J., 2010. Adaptive networks: coevolution of disease and topology. Phys. Rev. E 82 (3), 036116.

Meyers, L.A., 2007. Contact network epidemiology: Bond percolation applied to infectious disease prediction and control. Bull. Am. Math. Soc. 44 (1), 63–86.

Miller, J.C., 2009. Spread of infectious disease through clustered populations. J. R. Soc. Interface 6 (41), 1121–1134.

Miller, J.C., Slim, A.C., Volz, E.M., 2012. Edge-based compartmental modelling for infectious disease spread. J. R. Soc. Interface 9 (70), 890–906.

Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., Alon, U., 2002. Network motifs: simple building blocks of complex networks. Science 298 (5594), 824–827.

Newman, M.E.J., 2002. Spread of epidemic disease on networks. Phys. Rev. E 66 (1), 016128.

Newman, M.E.J., 2003. The structure and function of complex networks. SIAM Rev. 45 (2), 167–256.

Newman, M.E.J., Strogatz, S.H., Watts, D.J., 2001. Random graphs with arbitrary degree distributions and their applications. Phys. Rev. E 64 (2), 026118.

Nöel, P.-A., Davoudi, B., Lessard, S., Dubé, L.J., Allard, A., 2009. Time evolution of epidemic disease on finite and infinite networks. Phys. Rev. E 79 (2), 026101.

Pellis, L., House, T., Keeling, M.J., 2015. Exact and approximate moment closures for non-Markovian network epidemics. J. Theor. Biol. 382, 160–177.

Rand, D.A., 1999. Correlation equations and pair approximations for spatial ecologies. Advanced Ecological Theory: Principles and Applications. Blackwell Science, Oxford.

Read, J.M., Keeling, M.J., 2003. Disease evolution on networks: the role of contact structure. Proc. R. Soc. Lond. Ser. B: Biol. Sci. 270 (1516), 699–708.

Ritchie, M., Berthouze, L., House, T., Kiss, I.Z., 2014. Higher-order structure and epidemic dynamics in clustered networks. J. Theor. Biol. 348, 21–32.

Rogers, T., 2011. Maximum-entropy moment-closure for stochastic systems on networks. J. Stat. Mech.: Theory Exp. 2011 (5), P05007.

Rozhnova, G., Nunes, A., 2009. Fluctuations and oscillations in a simple epidemic model. Phys. Rev. E 79 (4), 041922.

Salathe, M., Jones, J.H., 2010. Dynamics and control of diseases in networks with community structure. PLoS Comput. Biol. 6 (4), e1000736.

Shirley, M.D.F., Rushton, S.P., 2005. The impacts of network topology on disease spread. Ecol. Complex. 2 (3), 287–299.

Trapman, P., 2007. On analytical approaches to epidemics on networks. Theor. Popul. Biol. 71 (2), 160–173.

Volz, E.M., 2008. SIR dynamics in random networks with heterogeneous connectivity. J. Math. Biol. 56 (3), 293–310.

Volz, E.M., Miller, J.C., Galvani, A., Ancel Meyers, L., 2011. Effects of heterogeneous and clustered contact patterns on infectious disease dynamics. PLoS Comput. Biol. 7 (6), e1002042.

Watts, D.J., Strogatz, S.H., 1998. Collective dynamics of 'small-world' networks. Nature 393 (6684), 440–442.
