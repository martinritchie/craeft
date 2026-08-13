infected by an external source, respectively. Summarizing all the above yields the complete system of equations,

$$\frac{dS(t)}{dt} = -\frac{\beta S(t)}{n} \theta(t),$$

$$\frac{d\theta(t)}{dt} = -\frac{\beta S(t)}{n} (\theta(t) - y(t)),$$

$$\frac{4M_{[02]}}{dt} = -2\lambda_2 \theta M_{[02]},$$

$$\frac{4M_{[12]}}{dt} = -M_{[12]}(\sigma + \gamma) + 2\lambda_2 M_{[02]} - \lambda_1(t) M_{[12]},$$

$$R(t) = 1 - S(t) - I(t).$$

This concludes the derivation for PDF-based epidemic dynamics on random networks. Volz et al. (2011) extended this framework to directed networks by defining a joint probability distribution which describes the typical number of links and triangles allocated to nodes. This particular derivation closely followed from this paper. However, we note that this framework can further be extended to hypergraph models where the joint probability specifies the distribution of subgraphs of various types around nodes. We refer to next to mean to read (PGFs). In Appendix A.1 we show that the PGF which is the main result of this paper can be made equivalent to the PGF resulting from Volz et al.'s original edge-triangle model.

## 2.2 Hypergraph configuration model

In this paper we generalize the configuration model (Bollobás 1980) to the hypergraph configuration model. Before we specify the model we need to establish how to classify hyperedges, the set of arcs that connect a node to all of the other members of a particular subgraph and their role within that subgraph.

To generate a hypergraph configuration model network one needs to first decide on the subgraph types in the network. We extend the configuration model (Bollobás 1980) by the identification of the number of different hyperedges indicated by the subgraphs. Hyperedges must be uniquely associated with both their parent subgraph and the role of their incident nodes (Karrer and Newman 2010) where the orbit of a node is the set of nodes with which it may be permuted such that no edges are created or destroyed. For example, in Fig. 1, subgraph $G_2$ contains two distinct orbits $\{r_{21}, r_{21}\}$ and $\{r_{22}, r_{22}\}$.

Once all hyperedges have been identified it is possible to define a joint probability distribution that specifies the probability of a node having a certain combination of these. Formally, if $P(s_1, s_2, \ldots, s_p)$ denotes the probability of a node having hyperedge sequence $\{s_1, s_2, \ldots, s_p\}$ we use the PGF framework to work with these degree sequences. For network generation these sequences will be subject to cardinality constraints. For instance, stub counts per orbit per node type must be divisible by three. Otherwise, the sequence needs to be re-generated. For asymmetric subgraphs,
