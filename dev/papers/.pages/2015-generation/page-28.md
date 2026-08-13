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
