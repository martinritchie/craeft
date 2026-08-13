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
