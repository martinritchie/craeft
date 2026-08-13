# Generation and analysis of networks with a prescribed degree sequence and subgraph family: higher-order structure matters

*Journal of Complex Networks* (2017) **5**, 1–31
doi:10.1093/comnet/cnw006
Advance Access Publication on 16 May 2016

Martin Ritchie

*School of Mathematical and Physical Sciences, Department of Mathematics, University of Sussex, Falmer, Brighton BN1 9QM, UK*

Luc Berthouze

*Centre for Computational Neuroscience and Robotics, University of Sussex, Falmer, Brighton BN1 9QH, UK*

and

Istvan Z. Kiss†

*School of Mathematical and Physical Sciences, Department of Mathematics, University of Sussex, Falmer, Brighton BN1 9QM, UK*

†Corresponding author. Email: I.Z.Kiss@sussex.ac.uk

Edited by: Ernesto Estrada

[Received on 26 November 2015; accepted on 9 March 2016]

Designing algorithms that generate networks with a given degree sequence while both varying subgraph composition and distribution of subgraphs around nodes is an important but challenging research problem. Current algorithms lack control of key network parameters, the ability to specify to what subgraphs a node belongs to come at a considerable complexity cost or, critically and sample from a limited ensemble of networks. To enable controlled investigations of the impact and role of subgraphs, especially for epidemics, we need algorithms that generate networks with a given degree sequence with the subgraph-level structural variants as diverse as possible. In this article, we present two new network generation algorithms that use subgraphs as building blocks to construct networks preserving a given degree sequence. Additionally, these algorithms provide control over clustering both at node and global level. In both cases, we show that being constrained by a degree sequence and global clustering results in topologies where degree and subgraph distributions and correlations are restricted. We suggest that for small- and large-scale network structure metrics such as path length and betweenness measures. Simulations of standard epidemic and complex contagion models on these networks show clearly that degree distributions and global clustering do not accurately predict the outcome of dynamical processes taking place on them. We conclude by discussing the benefits of the subgraph-centric approach.

*Keywords*: networks; clustering; subgraphs; epidemics; complex contagion.

## 1. Introduction

Being able to replicate, and therefore investigate, the structure and function of real-world complex networks is a profoundly difficult problem. However, the pervasiveness of a system that could be more
