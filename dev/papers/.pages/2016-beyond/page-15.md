The final step to generating the full system is to set the initial conditions. Only the initial conditions for subgraph states need resetting (from Fig. 3(b) and 4 (b) are fixed as per the previous section. This can be done by cycling through each element of **G**. If  $(G_a)$  is a purely susceptible state then we set $G_a = \rho$ and if $(G_a)$ contains a single infectious individual and is otherwise susceptible, we set $G_a = \mathcal{E}\rho$. All other states are set to zero, as we assume that with a sufficiently small infectious seed, the probability of having two infectious individuals in a subgraph is zero.

## 4 Results

To validate the proposed mean-field model and to assess the goodness of the approximation, we compare results from the ODEs to output from stochastic simulations. Networks were generated following the configuration algorithm, please refer to Appendix 5. Typically we considered 50 000 networks each with 15 000 and completed a single realisation of the epidemic, according to the Gillespie algorithm with the per link rate of infection $\tau = 0.5$ and recovery rate $\gamma = 1$, respectively, which leads to an effective reproductive number $R_0 = \tau \langle k \rangle / \gamma$. The initial conditions were set to 5 infectious individuals and an outbreak was said to occur if 5% infectious prevalence was observed. In all plots, the error bars for the ODE models and simulations denote bold lines and discrete points, respectively.

To start, we test the performance of our model against existing or state of the art models. To do this, in Fig. 7, we show results for two degree distributions that are homogeneous in the classical sense. Their PDFs are given by

$$g(k) = \frac{1}{2}\left(a_1 + 1\right)\binom{k}{a_1}$$

$$g(k) = \frac{1}{4}\left(a_1 + a_2 + a_3 + a_4\right)$$

where the variables $a_i$ correspond to subgraphs given in Fig. 1. Figure 7 shows that our results are in accord with theory [10,11] for Poisson and regular graphs. This is unsurprising but does provide a sanity check. In this figure, the clustering/transitivity ratios were measured following a recently developed subgraph algorithm [24], where the subgraph counts are defined as the ratio of a given subgraph count to all open and closed paths of length four, both counted uniquely. Currently, this specific clustering coefficient for the configuration algorithm does not information about the degree distribution, but does assume random mixing of subgraphs.

All models perform well in capturing the epidemic dynamics on networks generated using the distributions (e.g., see Fig. 7 and Table 1 for the peak, final size and timing). In Fig. 8, we apply a bimodal degree distribution and find that our model struggles to accurately capture the dynamics, both anticipating and compressing the epidemic's time frame, and over-estimating the final epidemic size (data not shown). The pairwise model does not encode any information relating to degree or
