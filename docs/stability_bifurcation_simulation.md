# Bates crash model: stability, bifurcation, and simulation

Code: `src/crash_model.py`. Figures: `figures/s1` to `figures/s5`. All numbers below come from one run of the script on the S&P 500 sample (19,319 daily returns, 4 January 1950 to 16 October 2026) with seed 20261106.

## 0. Model and parameters

The non-dimensional model from the previous stage is

$$
\begin{aligned}
dX &= \Big(m - \tfrac{s}{2}V\Big)\,d\tau + \sqrt{V}\,dB^{(1)} + Z\,dN, \qquad Z \sim N(a, b^2),\\
dV &= \varepsilon(1 - V)\,d\tau + \sqrt{2\varepsilon\delta V}\,dB^{(2)}, \qquad dB^{(1)}dB^{(2)} = \rho\,d\tau,
\end{aligned}
$$

where $\tau$ counts trading days, $X$ is log price in units of one day's long-run standard deviation $s$, $V$ is variance in units of its long-run mean, and $N$ jumps with probability $\Lambda$ per day. A simulated daily log return is $r = s\,\Delta X$.

| Group | Value | Meaning |
|---|---|---|
| $s$ | 0.0100 | one day's long-run σ |
| $m$ | 0.032 | daily Sharpe ratio |
| $\varepsilon$ | 0.0111 | variance reversion per day |
| $\delta$ | 1.0 | variance of $V$ |
| $\rho$ | −0.43 | price-variance shock correlation |
| $\Lambda$ | 0.00197 | jump probability per day |
| $a$, $b$ | −5.42, 1.84 | jump mean and standard deviation, in daily σ |

GBM sets $\delta = \Lambda = 0$, Merton sets $\delta = 0$, and Heston sets $\Lambda = 0$.

## 1. Linear stability of the deterministic skeleton

With noise and jumps removed, the substitutions $\tilde X = X - (m - s/2)\tau$ and $u = V - 1$ give the linear system

$$
\frac{d}{d\tau}\begin{pmatrix}\tilde X\\ u\end{pmatrix} = J\begin{pmatrix}\tilde X\\ u\end{pmatrix}, \qquad J = \begin{pmatrix}0 & -s/2\\ 0 & -\varepsilon\end{pmatrix}.
$$

$J$ is upper triangular, so its eigenvalues are $\lambda_1 = 0$ and $\lambda_2 = -\varepsilon = -0.0111$. Eliminating $u$ gives the second-order form used in the course notes, $\tilde X'' + \varepsilon\tilde X' = 0$, with characteristic equation $r(r + \varepsilon) = 0$ and the same roots.

The equilibria form the line $u = 0$ (the variance at its long-run value, any price level).

- **Variance direction.** $\lambda_2 < 0$, so $u(\tau) = u_0 e^{-\varepsilon\tau}$ decays to zero from every starting value. The half-life is $\ln 2/\varepsilon = 62.4$ trading days. The variance equation is linear, so this stability is global.
- **Price direction.** $\lambda_1 = 0$, so the price level is neutrally stable. A perturbation of $\tilde X$ persists indefinitely, which is the random-walk property of prices: the model has no restoring force towards a "fair" level.
- **Coupling.** The exact solution $\tilde X(\tau) = \tilde X(0) - \dfrac{s}{2\varepsilon}u_0\big(1 - e^{-\varepsilon\tau}\big)$ shows that a variance shock moves the price permanently by $-s u_0/(2\varepsilon) = -0.45\,u_0$ daily σ. A doubling of volatility ($u_0 = 3$) costs 1.35 daily σ, about 1.35 per cent, through the Itô term alone.

Figure `s1_phase_portrait.png` shows the trajectories, with $s$ enlarged to 0.2 so that the drift along $\tilde X$ is visible.

## 2. Stochastic stability

### 2.1 Moments

Itô's formula applied to $V^n$ gives a closed triangular system for $M_n = \mathbb{E}[V^n]$:

$$
\frac{dM_n}{d\tau} = -n\varepsilon M_n + n\varepsilon\big(1 + (n-1)\delta\big)M_{n-1}, \qquad M_0 = 1.
$$

The eigenvalues are $-\varepsilon, -2\varepsilon, \dots, -n\varepsilon$, all negative for every $\delta$, so every moment of $V$ converges. The stationary moments are $M_n = \prod_{j=0}^{n-1}(1 + j\delta)$, which equal the moments of $\mathrm{Gamma}(1/\delta, \delta)$; at $\delta = 1$ the script returns 1, 2, 6, 24. The $n$-th moment relaxes $n$ times as fast as the mean.

### 2.2 Limits of linearisation

Linearising the variance equation about $V = 1$ replaces $\sqrt{V}$ by 1 and gives an Ornstein-Uhlenbeck process with stationary law $N(1, \delta)$. This has the correct mean and variance and the wrong shape.

| $\delta$ | P($V<0$), linear | P($V>3$), exact | P($V>3$), linear | P($V<0.1$), exact | P($0<V<0.1$), linear |
|---|---|---|---|---|---|
| 0.05 | 3.9 × 10⁻⁶ | 6.4 × 10⁻¹⁰ | 1.9 × 10⁻¹⁹ | 6.4 × 10⁻¹⁴ | 2.5 × 10⁻⁵ |
| 0.25 | 0.023 | 0.0023 | 3.2 × 10⁻⁵ | 7.8 × 10⁻⁴ | 0.013 |
| 0.57 | 0.093 | 0.022 | 0.0040 | 0.026 | 0.024 |
| 1.0 | 0.159 | 0.050 | 0.023 | 0.095 | 0.025 |
| 2.0 | 0.240 | 0.083 | 0.079 | 0.248 | 0.023 |

At the screening values $\delta = 0.57$ to $1$, the linear model assigns 9 to 16 per cent probability to negative variance and understates the chance of a tripled variance by a factor of 2 to 6. High-variance states drive large falls, so the analysis of crashes uses the exact law throughout.

## 3. Bifurcations

The skeleton has a single equilibrium structure for all parameter values, so it has no deterministic bifurcation. The model has two phenomenological (P-) bifurcations, in which the shape of a stationary distribution changes qualitatively at a critical parameter value.

### 3.1 Variance density at $\delta = 1$

The zero-flux stationary solution of the Fokker-Planck equation for $V$ is

$$
p(V) \propto \frac{1}{2\varepsilon\delta V}\exp\!\left(\int \frac{2\varepsilon(1-V)}{2\varepsilon\delta V}\,dV\right) \propto V^{1/\delta - 1}e^{-V/\delta},
$$

a $\mathrm{Gamma}(1/\delta, \delta)$ density. Its mode is $1 - \delta$ for $\delta < 1$ and $0$ for $\delta \ge 1$.

| Regime | Density at $V \to 0$ | Most likely variance | Boundary $V = 0$ |
|---|---|---|---|
| $\delta < 1$ | 0 | $1 - \delta > 0$ | unreachable |
| $\delta = 1$ | 1 (exponential) | 0 | critical |
| $\delta > 1$ | $\infty$ | 0 | reachable |

The threshold $\delta = 1$ is the Feller condition $2\kappa\theta = \xi^2$ in dimensional form. For $\delta > 1$ the market spends most of its time at variance below its mean and reaches its mean variance through rare bursts, which is volatility intermittency. The screening range for $\delta$ (0.57 from 252-day kurtosis, 3.8 to 3.9 from daily kurtosis and the GARCH variance) lies on both sides of this threshold, so the data do not fix the regime. Figure `s2_variance_bifurcation.png` shows the densities and the mode as a function of $\delta$.

### 3.2 Crash mode in the one-day return density

With $V$ frozen over one day, the one-day density is

$$
f(x) = (1-\Lambda)\,\mathbb{E}_V\big[\varphi(x;\,0,\,V)\big] + \Lambda\,\mathbb{E}_V\big[\varphi(x;\,a,\,V + b^2)\big],
$$

where $\varphi(x;\mu,\sigma^2)$ is the normal density. When the jump mean $a$ is far enough into the loss tail, $f$ acquires a second local maximum: a crash mode, i.e. a typical crash size distinct from an extreme ordinary day. The script finds the critical jump mean $a^*$ by bisection on the number of modes ($b = 1.84$).

| $\Lambda$ | $a^*$, $\delta = 0$ | $a^*$, $\delta = 0.57$ | $a^*$, $\delta = 1$ |
|---|---|---|---|
| 10⁻⁴ | −5.63 | −8.61 | −9.67 |
| 10⁻³ | −5.17 | −7.31 | −8.04 |
| 0.00197 (screening) | −5.02 | −6.92 | −7.56 |
| 10⁻² | −4.65 | −5.99 | −6.41 |
| 10⁻¹ | −4.03 | −4.59 | −4.71 |

The density is bimodal for $a < a^*$. At the screening values ($a = -5.42$) Merton is bimodal and Heston-based mixing with $\delta \ge 0.57$ is unimodal. Stochastic variance thickens the ordinary tail, which fills the gap between the central mode and the jump mode and moves $a^*$ by 1.9 to 2.5 daily σ. A larger jump rate lowers the barrier: at $\Lambda = 0.1$ all three curves lie within 0.7 of one another. The sample contains about $\Lambda N \approx 38$ jump days, too few to resolve a second mode empirically, so in this model the question of whether crashes form their own class reduces to the position of $(\Lambda, a)$ relative to the curve. Figure `s3_crash_mode_bifurcation.png` shows the curves and four example densities.

## 4. Simulation

### 4.1 Scheme

- **Variance.** Exact CIR transition: over one day $V_{n+1}/c$ is non-central $\chi^2$ with $2/\delta$ degrees of freedom and non-centrality $V_n e^{-\varepsilon}/c$, where $c = \delta(1 - e^{-\varepsilon})/2$. This keeps $V > 0$ for every $\delta$, including $\delta > 1$, where an Euler step produces negative variance.
- **Log price.** Central scheme with $\bar V = (V_n + V_{n+1})/2$:

$$
\Delta X = \Big(m - \tfrac{s}{2}\bar V\Big) + \frac{\rho}{\sigma_v}\big(V_{n+1} - V_n - \varepsilon(1 - \bar V)\big) + \sqrt{1-\rho^2}\,\sqrt{\bar V}\,W + J\,\mathbf{1}_{\{U < \Lambda\}},
$$

with $\sigma_v = \sqrt{2\varepsilon\delta}$, $W \sim N(0,1)$, $U \sim \mathrm{Uniform}(0,1)$, and $J \sim N(a, b^2)$. The second term recovers the Brownian increment that drove $V$, which is how the leverage correlation enters.
- **Deterministic part.** The scheme is a one-day step of a linear relaxation with rate $\varepsilon$, which is stable for step lengths below $2/\varepsilon = 180$ days.
- **Runs.** 200 independent paths of 19,319 days per model, each started from the stationary variance law. The full script, including the bifurcation scans, takes about one minute.

### 4.2 Results

Medians over 200 paths, with the 5th to 95th percentile range in brackets.

| Statistic | GBM | Merton | Heston | Bates | S&P 500 |
|---|---|---|---|---|---|
| Annualised σ, % | 15.9 | 16.4 | 15.6 | 16.3 | 15.8 |
| Skewness | 0.00 | −0.37 | −0.11 | −0.49 [−0.67, −0.30] | −0.95 |
| Excess kurtosis | 0.0 | 2.5 | 2.9 | 5.1 [3.9, 6.9] | 25.2 |
| ACF of $\lvert r\rvert$, lag 1 | 0.00 | 0.00 | 0.27 | 0.24 [0.21, 0.27] | 0.27 |
| ACF of $\lvert r\rvert$, lag 100 | 0.00 | 0.00 | 0.077 | 0.072 [0.043, 0.102] | 0.094 |
| corr$(r_t, \lvert r_{t+1}\rvert)$ | 0.00 | 0.00 | −0.035 | −0.032 [−0.048, −0.017] | −0.108 |
| Days with $z < -3$ | 26 | 50 | 150 | 162 [141, 179] | 155 |
| Days with $z < -5$ | 0 | 20 | 8 | 28 [19, 36] | 34 |
| Falls of 5% or more | 0 | 20 | 6 | 27.5 [19, 37] | 30 |
| 3% falls recurring within 5 days | 0.00 | 0.02 | 0.19 | 0.17 [0.09, 0.25] | 0.42 |
| Worst day, % | −3.9 | −9.4 | −6.4 | −9.3 [−10.9, −8.1] | −20.5 |

Bates places the observed counts of 3σ falls, 5σ falls, and 5 per cent falls inside its 90 per cent simulation bands, and it is the single specification that does so for all three. Heston reproduces the 3σ count and volatility clustering and gives a quarter of the 5σ falls. Merton gives 60 per cent of the 5σ falls and no clustering. GBM gives none of them.

Four statistics lie outside every band:

1. **Kurtosis.** The highest 95th percentile is 6.9 against 25.2 observed (11.4 without 19 October 1987). The 1987 fall of 20.5 per cent is 1.9 times the 5th-percentile simulated worst day of 10.9 per cent.
2. **Leverage.** The model gives −0.03 against −0.11. With $\varepsilon = 0.011$, one day's variance change is small, so the price-variance correlation $\rho = -0.43$ transmits a weak next-day effect. Reproducing −0.11 needs a larger one-day variance response, through $\rho$ close to −1 or a larger $\delta$.
3. **Recurrence of large falls.** The model gives 17 per cent against 42 per cent. Bunching of 3 per cent falls within five days needs variance that rises within days, which a single factor with a 62-day half-life does not provide. The decay of the $\lvert r\rvert$ autocorrelation points the same way.
4. **Negative skew.** The model gives −0.49 against −0.95.

Each of these points to the same extension: a second, fast variance factor, or jump arrivals that depend on the current variance (self-exciting jumps), so that a crash raises the probability of the next one. Figures `s4_simulated_paths.png` and `s5_model_vs_data.png` show one path per model and the comparison with the data.
