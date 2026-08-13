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
