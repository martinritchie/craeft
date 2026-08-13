14 M. RITCHIE ET AL.

![Three unconnected subgraphs with nodes arranged in triangular and polygonal structures, illustrating the subgraph generation constraints described in Section 2.2; the subgraph of triangle (A,B,C) and triangle (D,E,F) results in three annotated distinct tours (A,B,C,D), (B,C,E) and (D,C,F) overlapping on one unintended triangle (C,F,D).](figure)

FIG. 3. Unintended generation of subgraphs with overlap. Despite satisfying the generation constraints given in Section 2.2, the subgraph of triangle $(A,B,C)$ and triangle $(D,E,F)$ results in three unintended distinct tours $(A,B,C,D)$, $(B,C,E)$ and $(D,C,F)$ overlapping on one unintended triangle $(C,F,D)$.

## 2.5. *Models of contagion*

In order to illustrate the impact of network structure—and higher-order structure particularly—different epidemic dynamics were simulated on the generated networks. Three different models were chosen: susceptible-infected-susceptible (SIS), SIR and complex contagion [30, 31]. To simulate SIS and SIR dynamics, the fully susceptible network of nodes is perturbed by infecting a small number of nodes. Infected nodes spread the infection to susceptible neighbours at a per link rate of infection $\tau$. Infected nodes either recover and become susceptible again (for SIS epidemics) at per node rate of recovery $\gamma$ (γ) or become removed (for SIR epidemics). In contrast to the infection process in the previous two dynamics, the complex contagion process requires that susceptible nodes are exposed to multiple infection events before becoming infected. These events must be from different infectious neighbours as only the first infection attempt from an infectious node counts. This critical infection threshold for each node is set in advance and is usually bounded from above by the degree of the node. To simulate the complex contagion dynamics, nodes are allocated infection thresholds $r_i \in \mathbb{N}$, where $i = 1, 2, \ldots, N$, and the fully susceptible network of nodes is then perturbed by infecting a small number of initial nodes. Under this model a susceptible node $i$ becomes infected as soon as it has received at least $r_i$ infectious contacts from $r_i$ distinct infected neighbours. There is no recovery in this model and infected individuals remain infected for the duration of the epidemic.

## 3. Results

### 3.1 *Algorithm validation*

To validate our algorithms, we generated a number of networks with pre-specified degree distribution and subgraph set, as well as a multinomial distribution of subgraph corners or hyperstubs around nodes. We verified that the networks generated were as expected given the input.
