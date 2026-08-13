From this PGF, the average number of subgraphs a node belongs to may be computed

$$\frac{\partial g(z)}{\partial z_i}\bigg|_{z=1} = \lambda_i = \langle s_i \rangle.$$

By replacing $z_i$ with $t^i$, where $i$ is the number of stubs contained within the hyperstub $h_i$, the PGF of the classical degree distribution can be recovered

$$g(t) := \exp\left(\sum_h \lambda_h(t_i^{c_1(h)} - 1)\right) = \exp\left(\lambda_1(t-1) + \lambda_2(t^2-1) + \cdots\right).$$

The $z^{th}$ term accounts for the fact that $G_0$ is counted twice, once for each of its hyperedges. The first and second moments of the degree distribution are directly computed using the linearity of expectation and the fact that $\text{Var}(X) = \sigma^2 X$. As well as recovering the degree distribution, it is possible to determine the expected number of triangles. Using $P(h_{\Delta}) = \lambda_{\Delta}$, the expected number on average each node in $G_0$ is incident to $3/2$ triangles. To summarise, we have

$$\langle k \rangle = \lambda_1 + 2\lambda_2 + \frac{3}{2}\lambda_{\Delta},$$
$$\text{Var}(k) = \lambda_1 + 4\lambda_2 + \frac{9}{4}\lambda_{\Delta},$$
$$\langle \Delta \rangle = \frac{3}{2}\lambda_{\Delta}.$$
$$(1)$$

By including a fourth subgraph in the above example, the equivalent of system Eq. (1) will be underdetermined with 3 equations and 4 unknowns. This allows the first and second moments and the expected number of triangles (clustering or clustering) to be fixed whilst varying the subgraph composition. For example, fixing $\langle k \rangle = 4, \text{Var}(k) = 5$ and $\langle \Delta \rangle = 2$, we can form the underdetermined system

$$\begin{pmatrix} 1 & 2 & 3/2 & 2 \\ 1 & 4 & 9/4 & 4 \\ 0 & 0 & 3/2 & 0 \end{pmatrix} \begin{pmatrix} \lambda_1 \\ \lambda_2 \\ \lambda_{\Delta} \\ \lambda_4 \end{pmatrix} = \begin{pmatrix} 4 \\ 5 \\ 2 \end{pmatrix},$$

where the columns of the LHS matrix correspond to contributions to $\langle k \rangle$, $\text{Var}(k)$ and $\langle \Delta \rangle$ respectively and $G_4$ denotes a complete subgraph of $r$ nodes. From this system it is possible to obtain the general solution: $(1) G_4 = P_4(t/2)$, and from a one-parameter family of solutions $\lambda_4 \in [0, \infty)$. By selecting different values of $\lambda_4$ and appropriately updating the LHS matrix, several differing network models with the same first and second moments and clustering may be obtained. A selection of such networks used in the results section is listed below:

Model 1 : $G_4 = P_4(t/2)$,
Model 2 : $G_4 = P_4(t/3)$,  $G_4 = P_4(t/2)$,
Model 3 : $G_4 = P_4(2t/3)$,  $G_4 = P_4(t/2)$,
