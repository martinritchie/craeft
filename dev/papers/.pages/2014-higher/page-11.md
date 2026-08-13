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
