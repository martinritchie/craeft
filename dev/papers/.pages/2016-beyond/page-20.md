The above sum considers each and every node from which an $s_i$ hyperstub originates. However, in hyperstub configuration model networks there is usually more than one type of hyperstub and this adds an additional level of detail to the excess degree. The calculations now may incorporate the different hyperstubs into its calculations. It is now possible to describe a nodes $s_i$ degree but conditioned on it being selected through one of its $s_j$ hyperstubs. More formally we can compute the expected excess degree using conditional expectation, $E(s_i|s_j = y)$, which yields

$$\tilde{s}_{j,i} = \frac{\sum_{s_j, s_i} s_i \frac{s_j}{\mu_{s_j}} P(s_j, s_i)}{\sum_{s_j, s_i} \frac{s_j}{\mu_{s_j}} P(s_j, s_i)}$$
(13)

where $\tilde{s}_{j,i}$ denotes the expected $s_i$ hyperstub degree observed from a node selected proportionally to its $s_j$ hyperstub degree. The denominator is given by Eq. (11), and the numerator is specified by

$$\sum_{s_j, s_i} s_i \frac{s_j}{\mu_{s_j}} P(s_j, s_i) = \frac{\langle s_j s_i \rangle}{\mu_{s_j}}.$$

## 6.2 ODEs for an example network

The following provides ODEs for a simple example network composed of only $G_0$ and $G_2$.

When deriving ODEs by hand listing out equations for $T_i$ is a good starting point as they include many of the subgraph states, i.e., $G_0(S)$, and can be used as the start of a check list when listing state equations.

$$T_2 = (G_0(S)1),$$
$$T_1 = (G_2(S)1),$$
$$T_3 = (G_0.(SS) + G_0(IS) + G_0(SI)) ,$$
$$T_4 = (G_2(SS) + G_2(IS) + G_2(SI)) ,$$
$$T_5 = (G_0.(SS) + G_0(IS) + G_0(SI)) ,$$
$$T_6 = (G_2(SS) + G_2(IS) + G_2(SI)) ,$$
$$T_7 = (G_0.(SSS) + G_0(ISS) + G_0(SIS) + G_0(SSI)) ,$$
$$T_8 = (G_2(SSS) + G_2(ISS) + G_2(SIS) + G_2(SSI)) .$$

It is important to note the above only lists states with $S$ in every subgraph state and that for a subgraph composed of $n$ will have $5^n$ state equations. For example, the first few state equations for $G_0$ are given by

$$\dot{G}_0(S) = -(T.A)G_0(SI) + (T.A)G_0(SI),$$
$$\dot{G}_0(SS) = -(\tau.A)G_0(ISS) - (\tau.A)G_0(SSI) + (\gamma.A)G_0(IS) + (\gamma.A)G_0(SS).$$
