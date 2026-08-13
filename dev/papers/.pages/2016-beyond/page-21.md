$$\dot{G}_0(IS) = -(\tau + \gamma)G_0(IS) - (T\Delta)_2 G_0(IS) + (T\Delta)_1 G_0(SS),$$

with equations for the following being omitted

$$\{\dot{G}_0(SR),\ \dot{G}_0(II),\ \dot{G}_0(IR),\ \dot{G}_0(RS),\ \dot{G}_0(RI),\ \dot{G}_0(RR)\},$$

Similarly, sample ODEs for the $G_\Delta$ subgraph, taken from a system of 27 ODEs, are:

$$\dot{G}_\Delta(SSS) = -[(T\Delta)_8 + (T\Delta)_4 + (T\Delta)_3]G_\Delta(SSS),$$
$$\dot{G}_\Delta(SSI) = -[2\tau + \gamma + (T\Delta)_4 + (T\Delta)_3]G_\Delta(SSI)$$
$$\quad\quad\quad\quad\quad + (T\Delta)_8 G_\Delta(SSS),$$
$$\dot{G}_\Delta(SIS) = -[2\tau + \gamma + (T\Delta)_8 + (T\Delta)_3]G_\Delta(SIS)$$
$$\quad\quad\quad\quad\quad + (T\Delta)_4 G_\Delta(SSS),$$
$$\dot{G}_\Delta(ISS) = -[2\tau + \gamma + (T\Delta)_8 + (T\Delta)_4]G_\Delta(ISS)$$
$$\quad\quad\quad\quad\quad + (T\Delta)_3 G_\Delta(SSS),$$

with equations for the following being omitted

$$\{\dot{G}_0(SSR),\ \dot{G}_0(SI I),\ \dot{G}_0(SIR),\ \dot{G}_0(SRS),\ \dot{G}_0(SRI),\ \dot{G}_0(SRR),$$
$$\dot{G}_0(ISI),\ \dot{G}_0(ISR),\ \dot{G}_0(IIS),\ \dot{G}_0(III),\ \dot{G}_0(IIR),\ \dot{G}_0(IRS),$$
$$\dot{G}_0(IRI),\ \dot{G}_0(IRR),\ \dot{G}_0(RSS),\ \dot{G}_0(RSI),\ \dot{G}_0(RSR),\ \dot{G}_0(RIS),$$
$$\dot{G}_0(RII),\ \dot{G}_0(RIR),\ \dot{G}_0(RRS),\ \dot{G}_0(RRR),\ \dot{G}_0(RRR)\}.$$

Each hyperstub will have a survivor function and a corresponding ODE describing its evolution, as follows:

$$\dot{\theta}_1 = -\theta_1 \frac{T_1}{M_1},$$
$$\dot{\theta}_2 = -\theta_2 \frac{T_2}{M_2},$$
$$\dot{\theta}_3 = -\theta_3 \frac{T_3}{M_3},$$
$$\dot{\theta}_4 = -\theta_4 \frac{T_4}{M_4},$$
$$\dot{\theta}_5 = -\theta_5 \frac{T_5}{M_5}.$$

The fraction of the population that is susceptible or infected is computed by compounding $\theta_i$ into the PGF. Symbolically, this is computed by the following

$$\dot{S} = \frac{d}{dt}\psi(\hat{\theta}),$$
$$\dot{I} = -\frac{d}{dt}\psi(\hat{\theta}) - \gamma I,$$
