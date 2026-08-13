of subgraph by-products can appear in addition to what was observed in the random networks depending on how one wishes to place the subgraphs around nodes.

We have seen that by using a modest selection of subgraphs, we have been able to substantially influence dynamics running on the network, particularly for SIR contagion dynamics. All results relative to this model indicate that constraining a network by degree sequence and clustering is not sufficient to accurately predict the outcome of the epidemic. More importantly, the results appear to suggest that the location of the critical regime depends on the higher-order structure of the network (above and beyond clustering).

We have also constructed networks with different numbers of prescribed triangles, which is certainly a key feature of any network construction algorithm. However, if such structural details do not impact on dynamics, their practical value is in doubt. By comparing models with similar structural properties with a limited set of network descriptors. Although degree sequence, degree–degree correlations and global clustering coefficient were observed to be the main drivers of disease transmission in models such as SIS and SIR, we found it not to be true in general. This is an important finding because one should remember that the dynamics simulated here are modest in complexity, when compared with models of neuronal dynamics for example, and yet, we were able to elicit significant differences by simply tuning the network structure above and beyond triangles. This implies that accounting for type and impact of higher-order structure may yet hold and reveal many important and surprising results.

## Acknowledgements

MR gratefully acknowledges Engineering and Physical Sciences Research Council (EPSRC, Doctoral Training Grant EP/K503198/1) and the University of Sussex for funding for his PhD. We would also like to thank Dr J.C. Miller for fruitful discussions on the pairwise contagion model [13], and for sharing his code for simulating the complex contagion model on networks [38].

## Appendix

### A.1 *Integer partitions*

The set of partitions of a positive integer $k$, lists all possible ways of writing $k$ as the sum of other positive integers. For example, the set of partitions of 4 is: $\{[4], [3, 1], [2, 2], [2, 1, 1], [1, 1, 1, 1]\}$. The number of ways to partition an integer is given by the partition function. For this derivation only we use $p(k)$ to denote the partition function of a positive integer $k$ and note that $p(1) = 1$, $p(2) = 2$, $p(3) = 3$, $p(4) = 5$, $p(5) = 7$, etc. For $k = 1, 2, 3, 4, 5$ ... the partition function returns $p(k) = 1, 2, 3, 5, 7, 11, \ldots$ respectively, and by convention $p(0) = 1$ and $p(-k) = 0$. This function can be used to calculate how many times a value for an integer $n < k$ appears in the partitions of $k$. We first compute the number of partitions in which $n$ will appear at least once. To determine the partitions we write $k$ as a partition in the following way: $[k - n, n]$ and use this notation to enumerate each part $n$ that appears in all remaining partitions of $k - n$. For example:

$$\{k = 4, n = 2\}:$$

$$[k - n + n] = [4],$$
$$[k - n + n] = [2 + 2] = [2, 2],$$
$$[k - n + n] = [1 + 1 + 2] = [1, 1, 2],$$
$$[k - n + n] = [3 + 1] = [3, 1], \tag{A.1}$$
