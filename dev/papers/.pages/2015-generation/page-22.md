22 M. RITCHIE ET AL.

TABLE 2. *Subgraph counts for network B* ($N = 5000, \lambda \sim \text{Pois}(5)$ *and* $\xi = 0.15$)*. The counts are unique. The expected counts are computed by summing the total counts from the subgraph sequences, dividing them by the subgraphs' node cardinality, and adding these figures to the number of subgraphs found as k-products in the random network. These are then subtracted by subgraphs that do not appear in any other subgraph.*

|          | $c4$ | $d4$ | $e4$ | $\langle 7 \rangle$ |
|----------|------|------|------|---------------------|
| Random   | 0    | 0    | 0    | 0                   |
| SDA      | 243  | 504  | 987  | 718                 |
| CMA      | 232  | 541  | 772  | 691                 |
| Expected | 243  | 504  | 619  | 741                 |

![Two plots side by side. Plot (a) shows SIS epidemic dynamics over time with a bell-shaped curve of infected nodes peaking around 4000 near time 40, with random network shown as dotted line and triangle markers, SDA as solid line with circle markers. Plot (b) shows SIR epidemic dynamics over time with a similar bell-shaped curve of infected nodes peaking around 1000 near time 15, then declining to zero, with the same marker conventions.](figure)

FIG. 11. (a) SIS and (b) SIR epidemic dynamics for network family B. The random, big-V and CDA lines have been plotted with a solid line, a circle and triangle markers, respectively. The SIS and SIR counts shown are the average of single Gillespie simulations on networks over 200 unique random, 200 SDA and 200 CMA networks. Both simulations were run using a single infected node as initial infectious seed of $I_0 = 10$ and had a per link rate of infection of $\tau = 1$ and recovered independently at rate $\gamma = 1$.

SIS and SIR Gillespie algorithms is available from https://github.com/martinritchie/dynamics. Accessed on 23 April 2016.

We know by construction that members of network family A were generated using different subgraphs, and Section 3.1 has shown that observable differences were found between networks in terms of average path length, betweenness centrality and subgraph composition. Despite this, Fig. 11, which show the time evolution for SIS and SIR dynamics, respectively, indicate that these dynamics can display a certain degree of insensitivity to these differences in structure. In this case, it is the SIR dynamics that show the greatest difference, in peak infectious prevalence (Fig. 11b) albeit quite marginal.
