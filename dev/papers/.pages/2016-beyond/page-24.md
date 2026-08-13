## 6.5 Algorithm 1: Hyperstub CM algorithm

**Algorithm 1:** The hyperstub configuration model. In this implementation, multiple edges are over written (line 33) but self-edges are permitted. To prevent this, if nodes already share an edge or a node has been selected twice (self-edge) lines 25-28 are repeated until a valid selection is made. This one-section step has been omitted below for readability.

1. **input** : $\mathcal{A}$
2. **output**: $G$

**Variables/initialisation**

3. Each row of $\mathcal{A}$ corresponds to single node's Hyperstub sequence.
4. Notation:
5. $a$: the hyperstub degree sequence, a non-negative $M \times H$ matrix;
6. $M$: the number of hyperedges/subgraphs;
7. $M$: the number of subgraphs;
8. $m$: the number of subgraphs;
9. $p$: the degree of a subgraph, $p \in \{1, 2, 3^+\}$;
10. $a$: the adjacency matrix of a subgraph, $p \in (0, 1)^{p \times p}$;
11. $n$: the number of nodes in $G$.

**Procedure**

12. The following creates dynamic lists (like 'hyperstub bins', one entry for each subgraph):
13. **for** each entry $h_i$ in $\mathcal{A}$ **do**
14. **for** each subgraph $p$ **do**
15. append (replicate of node $i$) to hyperstub bin($p$)
16. **end**
17. **end**
18. **for** each subgraph $p$ **do**
19. **for** each subgraph $m$ **do**
20. 1. Select uniformly at random and without replacement a subgraph incident to each desired hyperstub...
21. $v_m \leftarrow \text{rand}(\text{sample}(h_p))$
22. **end**
23. 2. The following compares pairs of the selected nodes...
24. 3. To determine their connectivity in A:
25. **for** $i \in \{0, 1, \ldots, p\}$ **do**
26. **for** $j \in \{0, 1, \ldots, p\}$ **do**
27. **if** $a[i, j] = 1$ **then**
28. add edge $(v_i, v_j)$ to $G$
29. **end**
30. **end**
31. **end**
32. **end**
