Model 4: $G_0 \sim \text{Poi}(3)$, $G_m \sim \text{Poi}(3)$.

While the three most basic network metrics for the networks above are identical, their degree distributions are not. However, it is also possible to generate classes of networks where the degree distribution is equal between networks but the subgraph composition is not. Let us consider networks composed entirely out of cycles, where, regardless of the length of the cycle, cycle hyperedges are composed of only pairs of stubs. It is then possible to increase the size of cycles while keeping many identical network metrics, including the degree distribution. The cycle networks are built in the following way: first, allocate to each node, on average, a pair of cycle hyperedges, then for each type of network allow the hyperedges to form increasingly large cycles starting with $G_0$, then $G_2$ and so on. If the hyperedges are distributed such that $k_i \sim \text{Poi}(2)$ then the overall degree distribution for our network will be such that only even degrees are possible, i.e., $P(\text{degree} = 2k) = P(\text{degree} = k)/P(\text{Poi}(2))$ denoted $G_c \sim 2 \text{Poi}(2)$ for conciseness. The networks we use to build cycle networks and, for comparison, i.e., a network with degree distribution given by $G_c \sim 2\text{Poi}(2)$ but connections random (without the restriction) will be using the following cycle based networks:

Null Model: $G_0 \sim 2\text{Poi}(2)$,

Model C1: $G_m \sim \text{Poi}(2)$,

Model C2: $G_0 \sim \text{Poi}(2)$,

Model C3: $G_2 \sim \text{Poi}(2)$,

Model C4: $G_3 \sim \text{Poi}(2)$.

where $G_2$ and $G_3$ denote cycles of 3 and 6 nodes (pentagons and hexagons), respectively. Having thus created two classes of networks, the former of which we can use to investigate the effects of increasing group size and the latter, as the same framework as far as dynamics are concerned, the latter to investigate the effect of cycles of increasing length on dynamics.

## 2.5 SIR epidemics on hypergraph configuration model networks

This section presents the derivation of a general *SIR* epidemic model for a network built from an arbitrary number of subgraph types. Conceptually, this model uses the pair-counting approach of Karrer and Newman (2010) and extends it to the AME framework of Volz et al. (2011), Volz (2008). By taking this approach it is possible to derive ODEs that accurately predict the spread of epidemics on networks that exhibit a variety of exotic subgraphs, both fully- and non-fully connected.

The first step is to choose the set of subgraphs to be included in the network. Let us suppose we have chosen $M$ subgraph types, $G_1, G_2, \ldots, G_M$. As an example, Fig. 3 shows $M = 5$ different subgraphs, which result in $m = 17$ distinct node positions, where $m$ stands for the number of distinct node positions, e.g., in a triangle, we count that a hyperedge is the set of half-links connecting a node to a subgraph. This example
