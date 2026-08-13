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
