**Table 4**
Combinations of $i$, $j$, $k$ and $l$ that satisfy (27).

| $i$ | $j$ | $k$ | $l$ |
|-----|-----|-----|-----|
| 1   | 2   | 3   | 4   |
| 1   | 3   | 2   | 4   |
| 1   | 4   | 2   | 3   |

In this form we see that it is possible to compute the individual counts using the following identities:

$$[\stackrel{\frown}{23}] = \sum_{i,j,k,l} \delta_{ij} \delta_{jk} \delta_{kl} \delta_{li} \delta_{lm} \delta_{mi}$$
(24)

$$[\stackrel{\frown}{23}] = \sum_{i,j,k,l} \delta_{ij} \delta_{jk} \delta_{kl} (1 - \delta_{il})$$
(25)

$$[\stackrel{\frown}{23}] = \sum_{i,j,k,l} \delta_{ij} \delta_{kl} (1 - \delta_{jk}) (1 - \delta_{il})$$
(26)

$\delta_{ij}\delta_{kl} = 1$ is counted 24 times. $[\stackrel{\frown}{23}]$ is counted 4 times and $[\stackrel{\frown}{23}]$ is counted 8 times, equal to the number of automorphisms associated with each motif type. By listing the different combinations of $\{i,j,k,l\}$ that satisfy (for diagonal squares):

$$\delta_{il}\delta_{jk}(1-\delta_{ij})(1-\delta_{ik}) = 1,\tag{27}$$

such that $i \neq j, k \neq l$ it is possible to gain insight into the cardinality of this count. Consider a diagonal square as orientated in the notation $[\stackrel{\frown}{23}]$ labelled starting at the top left node in a clock wise direction: $i$, $j$, $k$ and $l$. By listing the combinations in this way we have not included the trivial rotation symmetry (which goes to itself, the orbit between $j$ and $l$: the orbit between $i$ and $k$) and reflections (which includes both orbits) have been excluded.

Currently, based on an online and numerical tests, we conjecture that this is the only correct way scale-up from motif to multiplex motif counts. This method of counting is thorough but it would not be computationally reasonable size since it has complexity $O(N^4)$ for order-4 structures.

## References

Ball, F., Sirl, D., 2009. Stochastic SIR-type epidemics among a population partitioned into households and workplaces. Adv. Appl. Probab. 41 (1), 73–101.

Bansal, M., Bhosekar, A., Bhavanam, M., Iyengar, V., 2006 (supra). An Open-Source Software for Building and Analyzing Complex Networks. Available: http://cran.r-project.org/web/packages/igraph.

Barabasi, A.-L., Albert, R., 1999. Emergence of scaling in random networks. Science 286 (5439), 509–512.

Brouwers, L., Cakici, B., Camitz, M., Tegnell, A., Boman, M., 2010. Socially distanced infectious disease modeling of the 2009 H1N1 pandemic in Sweden. PLoS ONE 5 (9).

Cauchemez, S., Bhatt, S., et al., 2014. Unraveling the drivers of MERS-CoV transmission. Epidemics 6, 1–5.

Crofts, J.J., Higham, D.J., 2009. A weighted communicability measure applied to complex brain networks. J. R. Soc. Interface 6 (33), 411–414.

Danon, L., Read, J.M., House, T.A., Vernon, M.C., Keeling, M.J., 2012. Social encounter networks: characterising Great Britain. Proc. R. Soc. Lond. Ser. B: Biol. Sci. 280 (1765), 20131037.

Dorogovtsev, S.N., Goltsev, A.V., Mendes, J.F.F., 2008. Critical phenomena in complex networks. Rev. Mod. Phys. 80 (4), 1275.

Eames, K.T., Keeling, M.J., 2002. Modeling dynamic and network heterogeneities in the spread of sexually transmitted diseases. Proc. Natl. Acad. Sci. 99 (20), 13330–13335.

Erdős, P., Rényi, A., 1960. On the evolution of random graphs. Publ. Math. Inst. Hung. Acad. Sci. 5 (1), 17–60.

Estes, C., Abdel-Kareem, M., Bhowmik, D., 2012. Hepatitis C transmission: Epidemiologic considerations. Proceedings of the Royal Society of London, Series B: Biological Sciences. 280 (1765).

House, T., Keeling, M.J., 2011. Insights from unifying modern approximations to infections on networks. J. R. Soc. Interface 8 (54), 67–73.

House, T., Ross, J.V., Sirl, D., 2013. How big is an outbreak likely to be? Methods for epidemic final-size calculation. Proceedings of the Royal Society of London, Series B: Biological Sciences 280 (1765), 20122, 214.

Ilinskaya, O., Litovchenko, V., Otkidach, D., 2014. Clique-based algorithms for network motif discovery in temporal networks. Proceedings of the Royal Society of London, Series B: Biological Sciences.

Janssen, J., Laarraj, A., 2013. Spread of infection in directed and weighted social networks. BMC Syst. Biol. 7 (S), 74.

Jeong, H., Mason, S.P., Barabasi, A.-L., Oltvai, Z.N., 2001. Lethality and centrality in protein networks. Nature 411 (6833), 41–42.

Kemper, J.T., 1980. On the identification of superspreaders for infectious disease. Math. Biosci. 48 (1), 111–127.

Keeling, M.J., 1999. The effects of local spatial structure on epidemiological invasions. Proc. R. Soc. Lond. Ser. B: Biol. Sci. 266 (1421), 859–867.

Keeling, M.J., Eames, K.T.D., 2005. Networks and epidemic models. J. R. Soc. Interface 2 (4), 295–307.

Kermack, W.O., McKendrick, A.G., 1927. A contribution to the mathematical theory of epidemics. Proceedings of the Royal Society of London, Series A: Mathematical and Physical Sciences 115 (772), 700–721.

Klovdahl, A.S., 1985. Social networks and the spread of infectious diseases: the AIDS example. Soc. Sci. Med. 21 (11), 1203–1216.

Kucharski, A.J., Kwok, K.O., et al., 2014. The contribution of social behaviour to the transmission of influenza A in a human population. PLoS Pathog. 10 (6).

Lindquist, J., Ma, J., Van den Driessche, P., Willeboordse, F.H., 2011. Effective degree network disease models. J. Math. Biol. 62 (2), 143–164.

Lloyd-Smith, J.O., Schreiber, S.J., Kopp, P.E., Getz, W.M., 2005. Superspreading and the effect of individual variation on disease emergence. Nature 438 (7066), 355–359.

Manitz, J., Kneib, T., Schlather, M., Helbing, D., Brockmann, D., 2014. Origin detection during food-borne disease outbreaks — a case study of the 2011 EHEC/HUS outbreak in Germany. PLoS Curr. 6.

Marceau, V., Noël, P.-A., Hébert-Dufresne, L., Allard, A., Dubé, L.J., 2010. Adaptive networks: coevolution of disease and topology. Phys. Rev. E 82 (3), 036116.

Meyers, L.A., 2007. Contact network epidemiology: Bond percolation applied to infectious disease prediction and control. Bull. Am. Math. Soc. 44 (1), 63–86.

Miller, J.C., 2009. Spread of infectious disease through clustered populations. J. R. Soc. Interface 6 (41), 1121–1134.

Miller, J.C., Slim, A.C., Volz, E.M., 2012. Edge-based compartmental modelling for infectious disease spread. J. R. Soc. Interface 9 (70), 890–906.

Milo, R., Shen-Orr, S., Itzkovitz, S., Kashtan, N., Chklovskii, D., Alon, U., 2002. Network motifs: simple building blocks of complex networks. Science 298 (5594), 824–827.

Newman, M.E.J., 2002. Spread of epidemic disease on networks. Phys. Rev. E 66 (1), 016128.

Newman, M.E.J., 2003. The structure and function of complex networks. SIAM Rev. 45 (2), 167–256.

Newman, M.E.J., Strogatz, S.H., Watts, D.J., 2001. Random graphs with arbitrary degree distributions and their applications. Phys. Rev. E 64 (2), 026118.

Nöel, P.-A., Davoudi, B., Lessard, S., Dubé, L.J., Allard, A., 2009. Time evolution of epidemic disease on finite and infinite networks. Phys. Rev. E 79 (2), 026101.

Pellis, L., House, T., Keeling, M.J., 2015. Exact and approximate moment closures for non-Markovian network epidemics. J. Theor. Biol. 382, 160–177.

Rand, D.A., 1999. Correlation equations and pair approximations for spatial ecologies. Advanced Ecological Theory: Principles and Applications. Blackwell Science, Oxford.

Read, J.M., Keeling, M.J., 2003. Disease evolution on networks: the role of contact structure. Proc. R. Soc. Lond. Ser. B: Biol. Sci. 270 (1516), 699–708.

Ritchie, M., Berthouze, L., House, T., Kiss, I.Z., 2014. Higher-order structure and epidemic dynamics in clustered networks. J. Theor. Biol. 348, 21–32.

Rogers, T., 2011. Maximum-entropy moment-closure for stochastic systems on networks. J. Stat. Mech.: Theory Exp. 2011 (5), P05007.

Rozhnova, G., Nunes, A., 2009. Fluctuations and oscillations in a simple epidemic model. Phys. Rev. E 79 (4), 041922.

Salathe, M., Jones, J.H., 2010. Dynamics and control of diseases in networks with community structure. PLoS Comput. Biol. 6 (4), e1000736.

Shirley, M.D.F., Rushton, S.P., 2005. The impacts of network topology on disease spread. Ecol. Complex. 2 (3), 287–299.

Trapman, P., 2007. On analytical approaches to epidemics on networks. Theor. Popul. Biol. 71 (2), 160–173.

Volz, E.M., 2008. SIR dynamics in random networks with heterogeneous connectivity. J. Math. Biol. 56 (3), 293–310.

Volz, E.M., Miller, J.C., Galvani, A., Ancel Meyers, L., 2011. Effects of heterogeneous and clustered contact patterns on infectious disease dynamics. PLoS Comput. Biol. 7 (6), e1002042.

Watts, D.J., Strogatz, S.H., 1998. Collective dynamics of 'small-world' networks. Nature 393 (6684), 440–442.
