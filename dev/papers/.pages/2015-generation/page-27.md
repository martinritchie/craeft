i.e., there will be $p(k - a)$ such partitions. To more formally show this we use Euler's partition theorem

$$\sum_{n=0}^{\infty} p(n) z^n = \prod_{k=1}^{\infty} \frac{1}{1-z^k}$$
(A.2)

To calculate values of $p$, we first expand the r.h.s of the above

$$(1 + z + z^2 + \cdots)(1 + z^2 + z^4 + \cdots)(1 + z^3 + z^6 + \cdots) \cdots$$

such that to find the value of, e.g., $p(2)$, we simply collect the powers of $z^2$ to reveal the coefficient

$$p(2)z^2 = 2z^2.$$

In general, the terms in the geometric series in powers of $k$, i.e., $(1 + (z^k)^1 + (z^k)^2 + \cdots)$, give the number of times the integer $k$ may contribute to $n$ for the $z^n$ term, assuming that $k \leq n$. For example, $z^5$ can be formed by $(z^1)(z^4)^1$, which means that $5 = 2 + 2 + 1$ or by $(z^1)(z^2)^2$ which means that $5 = 1 + 1 + 1 + 2$, or $(z^1)(z^2)^2$ which means that $5 = 2 + 2 + 1$. To prove that $p(k - a)$ gives the number of partitions which $a$ appears at least once, we write the following

$$(1 + z + z^2 + \cdots)(1 + z^2 + z^4 + \cdots) \cdots (1 + z^{a-1} + \cdots) \cdots$$

where the exclusion of the $z^0$ term guarantees that the power of each and every term includes at least one alpha. Then to express the terms to the power of $n$ as

$$z^a + z^{2a} + \cdots = \frac{1}{1 - z^a} - 1$$
$$= \frac{z^a}{1 - z^a}$$
(A.3)

Euler's theorem can be modified to account for such a term

$$z^a \sum_{n=0}^{\infty} p(n) z^n = z^a \prod_{k=1}^{\infty} \frac{1}{1-z^k}$$
(A.4)

Comparing like-for-like powers in this modified expression gives the coefficient of $z^n$ as $p(n - a)$. Similarly, by writing

$$z^{ma}(1 + z^a + z^{2a} + \cdots) = z^{ma} \frac{1}{1 - z^a}$$

the result holds for multiples of $a$: $p(n - ma)$, the number of partitions in which $a$ appears at least $m$ times. Using the cumulative property of this expression, it is possible to compute the number of partitions in which $a$ appears exactly $m$ times

$$p(k - a(m-1)) - p(k - am)$$
