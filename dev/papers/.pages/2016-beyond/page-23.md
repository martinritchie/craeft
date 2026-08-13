where the following substitutions, $z_i = y/j$ and $a_i = a$, were made so that $a^s$ is the result of the above product. Now, every time an $s_i$ is allocated, we allocate an $s$ instead. Finally, since $p_Y$ is a joint distribution of $I$ identically distributed independent random variables, i.e., $\vec{y} = (y/j, y/j, \ldots, y/j)$, we get:

$$p_{\vec{y}}(a \to a) = \frac{a}{s} \sum_{j=1}^{s} p_s^{a^{j-1}}$$
(16)

It is also possible to translate between the two models elsewhere in the derivation. As an example, in our approach, infection over lines is given by $T_1$ and $T_2$ as per Eq. (3). By summing these values, the equivalent values used in Volz et al.'s formulation may be recovered. Following our derivation, first let $G_0(S) \equiv G_0(Z/S)$ and:

$$T_1 + T_2 = (G_0(S)/S) + (G_0(S) + G_0'(S))$$

Since each $G_0$ is generated from a PGF that allocates positions at rate 1/2 that of Volz et al.'s PGF, the 2 will cancel yielding $\tau G_0(S)$. However, it is only necessary to show equivalence between the two PGFs since all other variables follow from this.

## 4.4 State transition matrix

The state transition matrix for $G_0$ (lines) is given by:

$$Z = \begin{pmatrix} (IS) & (II) & (IR) & (SI) & (I) & (IS) & (RS) & (R) & (R) \\ (IS) & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ (II) & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ (IR) & 0 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ (I) & \tau/2 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\ (IS) & 0 & \tau/2 & 0 & 0 & 0 & 0 & 0 & 0 \\ (RS) & 0 & 0 & 0 & \gamma & 0 & 0 & 0 & 0 \\ (R) & 0 & 0 & \tau/2 & 0 & \gamma & 0 & 0 & 0 \\ (R) & 0 & 0 & 0 & 0 & 0 & \gamma & 0 & 0 \end{pmatrix}$$

$\hat{Z}$ Springer
