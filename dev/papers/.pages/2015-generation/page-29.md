## A.3 Pseudocode for CMA

**Algorithm 2:** Pseudocode for CMA. Other steps, such as ensuring the handshake lemma is satisfied for both lines and subgraphs, are identical to what is used for the UDA and are detailed in Section 2.1 and can be viewed in the Matlab source code. The output hyperstub degree sequence $H$ must be used as input for a modified configuration model procedure to realize a network; see Algorithm 3.

```
1  input : D = (d_1, d_2, ..., d_N), G = {G_1, G_2, ..., G_L}, Y = {S_1, S_2, ..., S_L}
2  output: H ∈ N^{G_N}
3  Variables
4  // D: degree sequence, N: number of nodes,
5  // G: set of subgraphs, L: number of subgraphs,
6  // S: subgraph sequence, g: subgraph adjacency matrix,
7  // y: multinomial proportions, h: hyperstubs in a subgraph, H: hyperstub degree sequence
8  Procedure
9  for Each subgraph, G_i do
10    % Identify the degree sequence, s, of the subgraph.
11    s_i = Σ_j g_{i,j}, s = unique(s), m = length(s)
12    % y reflects the proportions of hyperstubs
13    P_i = (P_{i,1}, ..., P_{i,m})
14    y_i = P_i/||P_i||
15    % The subgraph is decomposed into a hyperstub
16    % y corresponds to the multinomial distribution, M.
17    % so that H_i ∈ N^{s_{i,1} × ... × s_{i,m}}
18    H(j) = M(S(j), y_i)
19 end
20    H_i^' is a sequence of the true stub count
21    H_i^' = H_i · s_i
22    % Sum so that H_i^' ∈ N^{G_N}
23    H(j) = Σ_{i=1}^{L} H(i, j)
24 end
25 while elements of each H_i are non-zero do
26    % Find the largest subgraph degree.
27    [i, j] = Find the largest hyperstub degree.
28    % i.e., the i^th element of H_j
29    % Find all elements of the degree sequence at least this large and
30    % select an element from d at random
31    d' = {d ∈ D : d ≥ s_{i,j}}, m = d (random)
32    % Find E(j) to e' and update
33    % c's available stubs of index j (random)
34    λ = l = H(j), H(j) = 0
35 end
```
