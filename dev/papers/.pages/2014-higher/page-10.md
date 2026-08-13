![Plots of final endemic size and endemic equilibrium fractions versus the rate of r increasing from r=0.1 to r=1.0 in increments of 0.5. Dotted networks were generated via Gillespie simulations performed for each value of r. The networks were homogeneous with N=5 and k=1000.](figure)

computational complexity, it has the slight disadvantage that it does not provide the multiplying type of counting used in pairwise models. In the Appendix, we conjecture that this can be easily overcome by simply multiplying by *k*, and this can be confirmed numerically by the simulations performed here.

It has been demonstrated that care needs to be taken when trying to extend modelling to clustered networks. Whilst models for single clustered networks composed of exclusively non-overlapping triangles and edges have been developed, it is yet to be more meaningfully extended to topologies with a higher order of clustering. Networks such as a square with a diagonal or a fully connected square may fulfil some function depending on the area of application (e.g. genetic networks), and therefore developing methods that are capable of quantifying this correctly is crucial for further model development.

Many extensions for this work exist ranging from considerations around higher-order structure, algorithmic efficiency in measuring these and developing stochastic network models that allow clearer interpretation control of not only lower, but also higher-order structures.

## Acknowledgements

Martin Ritchie acknowledges funding for his PhD studies from EPSRC (Engineering and Physical Sciences Research Council) and the University of Sussex. Thomas House acknowledges funding from MRC and would like to thank Charo I. del Genio for discussions on the Motif Decomposition algorithm.

## Appendix A

### *A.1. Motif decomposition analysis*

It is possible to write down the dynamics for the SIS process (Section 6.1.1) in the limit of large networks by decomposing motifs into hyper motifs (considering each motif at a higher level of scale) and considering the links between them. We now consider the process being performed in a homogeneous network with *N* nodes and each node having exactly degree *k* and using the notation that $S_k = \langle S \rangle$ and $I_k = \langle I \rangle$. In this scenario, it is possible to write equations for the normalised count of each hypermotif:

$$\dot{S}_k = -\tau S_k I_k + \gamma I_k, \tag{13}$$

$$\dot{S}_{kk} = -\hat{\tau}(S_{kk}I_k + S_{kk}S_kI_{kk}/S_k^2) + \hat{\gamma}(S_{kI_k} - S_{kk}), \tag{14}$$

$$\dot{S}_{kI_k} = -\hat{\tau}(S_{kk}I_k + S_{kI_k}I_k - S_{kk}S_kI_{kk}/S_k^2 - S_{kI_k}^2/S_k) + \hat{\gamma}(I_{kI_k} - S_{kI_k} - S_{kI_k}), \tag{15}$$

$$\dot{I}_{kk} = -\hat{\tau}(I_{kk}I_k + I_{kk}I_k) + \hat{\tau}S_{kI_k} - \hat{\gamma}(I_{kk} + I_{kk}), \tag{16}$$

$$\dot{S}_{kk^2} = -\hat{\tau}(S_{kk^2}I_k + 2S_{kk^2}S_kI_{kk}/S_k^2) + \hat{\gamma}(2S_{kI_k^2} - S_{kk^2}). \tag{17}$$

These equations can be solved for initial conditions $S_k(0) = 0.9$ and the remaining fraction $I_k(0) = 0.1$ with re-noting a dot is just included for clarity and can be set to 1 with no loss of generality. The process stops at a time *t*\* when it is determined that clustering has been achieved:

$$T_c^1(k) = \frac{\phi}{k}, \tag{18}$$

where $T_c$ denotes the number of triangles associated with each node in the network. These equations can be solved for the motif structure and inserted into Eqs. (1)–(5) to obtain a prediction for motif structure. The hyper-motifs can be done for a star but quickly become non-intuitive. It is also possible to use the quantities in Table 1 to derive epidemic final size and other attributes. Reading
