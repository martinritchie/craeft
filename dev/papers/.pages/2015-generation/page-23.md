GENERATION AND ANALYSIS OF NETWORKS 23

![Two panels showing complex contagion dynamics for network family A. Left panel: probability over time (x-axis 10–50) for Random, Big-V, and UDA networks. Right panel: final size probability distribution (x-axis 1000–5000) for the same three network types.](figure)

FIG. 12. Complex contagion dynamics for network family **A**. The complex contagion epidemics we parameterised as initial infections seed of $I_0 = 250$ and a final threshold of infection of $r = 2$.

![Two panels showing SIS and SIR epidemic dynamics for network family B. Left panel: number infected over time (x-axis 0–6) for Random, UDA, and CMA networks, showing curves peaking around time 2–3. Right panel: similar dynamics over time (x-axis 0–500) showing final epidemic trajectories.](figure)

FIG. 13. (a) SIS and (b) SIR epidemic dynamics for network family **B**. The random, UDA and CMA data has been plotted with a thin line to show the spread of epidemic trajectories. The bold lines show the mean trajectory based on each of the 1000 network realisations from each network generation algorithm. The SIS and SIR epidemics were seeded with an initial infection seed of $I_0 = 10$ and had a per link rate of infection of $\beta = 1$ and recovered independently at rate $\gamma = 1$.

In contrast, complex contagion dynamics do show sensitivity to structural differences found between Big-V and UDA networks. Figure 12 reveals that for UDA networks the epidemic fully percolates in almost 100% of the simulations instead of only 80% of the cases for Big-V networks and that epidemics on UDA networks achieve this steady state in less time. This indicates that whilst UDA networks operate in the super critical regime, Big-V networks are closer to the transition point. Locating this transition is possible but is beyond the scope of this article.

When network family **A** is used, the networks' degree distribution and clustering appear to be the main determinants of the time evolution and outcome of the SIS and SIR epidemics. In contrast, when network family **B** is used, Figs. 13 and 14 show that all dynamics considered are impacted by differences in network
