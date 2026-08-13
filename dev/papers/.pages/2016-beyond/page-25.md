### 6.6 Algorithm 2 : Transition matrix algorithm

**Algorithm 2:** Generating the state transition matrix. The comparison in line 17 needs to check: (1) that only a single node has changed state and (2) only state changes $S \rightarrow I$ and $I \rightarrow R$ are valid.

```
1  input  : g,
2  output: Z.

3  Variables / initialisation
4  g: the adjacency matrix of a subgraph G,
5  % Z ∈ ℝ^{3^n × 3^n}.
6  Z: matrix corresponding rate of transition between states of G,
7  n: node count of G,
8  % G̃ contains 3^n elements.
9  G̃: the vector of states of G,
10 τ: per link infection rate,
11 γ: recovery rate,
12 T Δ: the expected force of infection a node within G experiences from outside G.

13 Procedure
14 for every state G̃_i do
15   for every state G̃_j do
16     % Compare each and every possible state transition of G:
17     switch G̃_i → G̃_j do
18       case A single infection occurs
19         if the new I is connected to another I within G then
20           % Check the connectivity of the new I using g.
21           Z_{i,j} = τ + T Δ
22         else
23           % the infection was from only an external source.
24           Z_{i,j} = T Δ
25         end
26       end
27       case A single recovery occurs
28         Z_{i,j} = γ
29       end
30       case otherwise
31         Z_{i,j} = 0
32       end
33     endsw
34   end
35 end
```
