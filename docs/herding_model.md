# RQ2: herding model

Code: `src/herding_model.py` (runs in about 8 seconds). Figures: `figures/h1` to `figures/h5`. All numbers below come from one run with seed 20261106.

| RQ2 item | Section | Status |
|---|---|---|
| Assumptions | 1 | complete |
| Conservation of traders | 2 | complete |
| Mean-field ODE | 3 | complete |
| Non-dimensionalisation | 4 | complete |
| Equilibria | 5 | complete |
| Stability | 6 | complete |
| Bifurcation | 7 | complete |
| Finite-N simulation | 8 | complete |

## 1. Assumptions

1. **Population.** A fixed number $N$ of traders, each either optimistic ($+$, buys) or pessimistic ($-$, sells). No trader enters or leaves.
2. **Switching.** Each trader reconsiders at a constant rate $\nu$, independently of the others. A pessimist becomes an optimist with rate $\nu e^{U}$ and an optimist becomes a pessimist with rate $\nu e^{-U}$.
3. **Herding.** The switching incentive is $U = \alpha x + \beta$, where $x = (n_+ - n_-)/N$ is the opinion index. $\alpha \ge 0$ measures how strongly traders copy the majority, and $\beta$ is an exogenous bias from news or fundamentals.
4. **Mean field.** Every trader observes the same aggregate $x$; there is no network structure.
5. **Price.** Fundamentalist traders absorb the net demand of the herding traders. Market clearing, $N T_c x + N_f T_f (p_f - p) = 0$, gives a log price deviation proportional to opinion, $p - p_f = \lambda x$ with $\lambda = N T_c/(N_f T_f)$. A change in $x$ maps to a price change $\lambda\,\Delta x$.

The exponential form follows Lux (1995). It keeps rates positive and makes the ratio of up and down rates $e^{2U}$, so a unit increase in $U$ multiplies the odds of switching to optimism by $e^2$.

## 2. Conservation of traders

Let $n_+(t)$ and $n_-(t)$ be the numbers of optimists and pessimists. Switching moves traders between the two groups and creates none:

$$
\frac{dn_+}{dt} = \underbrace{n_-\,\nu e^{U}}_{\text{inflow from }-} - \underbrace{n_+\,\nu e^{-U}}_{\text{outflow to }-}, \qquad \frac{dn_-}{dt} = -\frac{dn_+}{dt}.
$$

Adding the two equations gives $\dfrac{d}{dt}(n_+ + n_-) = 0$, so $n_+ + n_- = N$ for all $t$. Conservation reduces the system from two unknowns to one, $n_+$, or equivalently $x \in [-1, 1]$, with $n_\pm = N(1 \pm x)/2$.

At finite $N$ the count $n = n_+$ is a birth-death chain with transition rates

$$
W_+(n) = (N - n)\,\nu e^{U(n)}, \qquad W_-(n) = n\,\nu e^{-U(n)},
$$

and its probability $P(n, t)$ obeys the master equation

$$
\frac{\partial P(n,t)}{\partial t} = W_+(n-1)P(n-1,t) + W_-(n+1)P(n+1,t) - \big[W_+(n) + W_-(n)\big]P(n,t).
$$

This is the same conservation statement at the level of probability: the rate of change of $P(n,t)$ equals the probability flux into state $n$ minus the flux out.

## 3. Mean-field ODE

Multiplying the master equation by $n$ and summing gives $\dfrac{d\langle n\rangle}{dt} = \langle W_+(n) - W_-(n)\rangle$. Fluctuations of $x$ have size $O(N^{-1/2})$, so for large $N$ the average of the rates equals the rates at the average. Substituting $n_\pm = N(1 \pm x)/2$:

$$
\frac{dx}{dt} = \frac{2}{N}\frac{dn_+}{dt} = \nu\big[(1 - x)e^{U} - (1 + x)e^{-U}\big] = 2\nu\big[\sinh U - x\cosh U\big],
$$

$$
\boxed{\;\frac{dx}{dt} = 2\nu\cosh(\alpha x + \beta)\,\big[\tanh(\alpha x + \beta) - x\big]\;}
$$

The factor $\cosh U > 0$ sets the speed; the bracket sets the direction. $x$ rises when the fraction implied by the incentive, $\tanh U$, exceeds the current fraction $x$.

## 4. Non-dimensionalisation

| Quantity | Meaning | Dimension |
|---|---|---|
| $t$ | time | T |
| $\nu$ | reconsideration rate per trader | T⁻¹ |
| $x$, $\alpha$, $\beta$, $N$ | opinion index, herding strength, bias, population | 1 |

Time is the single dimension. With the time scale $1/(2\nu)$, $\tau = 2\nu t$, the ODE becomes

$$
\frac{dx}{d\tau} = F(x;\alpha,\beta) = \cosh(\alpha x + \beta)\big[\tanh(\alpha x + \beta) - x\big].
$$

The Buckingham $\Pi$ theorem gives three groups from the four quantities $t, \nu, \alpha, \beta$ in one dimension: $\nu t$, $\alpha$, and $\beta$. The mean-field dynamics therefore depend on two parameters, $\alpha$ and $\beta$, and $\nu$ sets the clock alone. $N$ enters at finite size (Section 8). If each trader reconsiders once every two days ($\nu = 0.5$ per day), then $2\nu = 1$ per day and one unit of $\tau$ is one trading day.

## 5. Equilibria

$F = 0$ requires $\tanh(\alpha x^* + \beta) = x^*$, because $\cosh U > 0$. Equilibria are the intersections of the line $y = x$ with the sigmoid $y = \tanh(\alpha x + \beta)$.

- **$\beta = 0$, $\alpha \le 1$.** The sigmoid has slope $\alpha \le 1$ at the origin and slope below $\alpha$ elsewhere, so $x^* = 0$ is the single equilibrium: a balanced market.
- **$\beta = 0$, $\alpha > 1$.** The sigmoid starts steeper than the line and flattens to $\pm 1$, so it crosses the line three times: $x^* = 0$ and $x^* = \pm x_h$ with $x_h = \tanh(\alpha x_h)$. The values $\pm x_h$ are the optimistic and pessimistic herding states.
- **$\beta \ne 0$.** The bias shifts the sigmoid sideways. For $\alpha > 1$ there are three equilibria when $|\beta| < \beta_c(\alpha)$ and one otherwise (Section 7.2).

| $\alpha$ | $\beta$ | $x^*$ |
|---|---|---|
| 0.5 | 0 | 0 |
| 1.5 | 0 | −0.8586, 0, 0.8586 |
| 2.0 | 0 | −0.9575, 0, 0.9575 |
| 1.5 | 0.1 | −0.8011, −0.2060, 0.8939 |
| 1.5 | 0.3 | 0.9359 |

## 6. Stability

Linearising about $x^*$ with $x = x^* + \xi$ gives $d\xi/d\tau = F'(x^*)\,\xi$, so $\xi \propto e^{F'(x^*)\tau}$ and $x^*$ is stable when $F'(x^*) < 0$. Writing $F = \sinh U - x\cosh U$,

$$
F'(x) = \alpha\cosh U - \cosh U - \alpha x \sinh U,
$$

and at an equilibrium $\sinh U^* = x^*\cosh U^*$, so

$$
\boxed{\;F'(x^*) = \cosh(U^*)\,\big[\alpha(1 - x^{*2}) - 1\big]\;}
$$

An equilibrium is stable when $\alpha(1 - x^{*2}) < 1$: the herding gain at that state, reduced by the factor $1 - x^{*2}$ as the population saturates, is below the restoring rate of 1.

- **Balanced state, $\beta = 0$.** $F'(0) = \alpha - 1$. It is stable for $\alpha < 1$ and unstable for $\alpha > 1$. At $\alpha = 1$, $F'(0) = 0$ and linearisation is inconclusive; the cubic term decides (Section 7.1).
- **Herding states, $\beta = 0$, $\alpha > 1$.** $g(x) = \tanh(\alpha x) - x$ is concave for $x > 0$ with $g(0) = 0$ and $g'(0) = \alpha - 1 > 0$, so it crosses zero at $x_h$ with $g'(x_h) < 0$, i.e. $\alpha(1 - x_h^2) < 1$. Both herding states are stable.

| $\alpha$ | $\beta$ | $x^*$ | $F'(x^*)$ | Stable |
|---|---|---|---|---|
| 0.5 | 0 | 0 | −0.500 | yes |
| 1.0 | 0 | 0 | 0 | marginal |
| 1.5 | 0 | 0 | 0.500 | no |
| 1.5 | 0 | ±0.8586 | −1.181 | yes |
| 2.0 | 0 | ±0.9575 | −2.890 | yes |
| 1.5 | 0.1 | −0.8011 | −0.773 | yes |
| 1.5 | 0.1 | −0.2060 | 0.446 | no |
| 1.5 | 0.1 | 0.8939 | −1.558 | yes |

The code checks each value of $F'(x^*)$ against direct differentiation of $F$; the two agree in every row. The recovery time from a small disturbance is $1/|F'(x^*)|$: 0.85 days in the herding state at $\alpha = 1.5$ against 2 days in the balanced state at $\alpha = 0.5$.

## 7. Bifurcations

### 7.1 Supercritical pitchfork at $\alpha = 1$, $\beta = 0$

Expanding $F$ for small $x$ with $\beta = 0$:

$$
\frac{dx}{d\tau} = (\alpha - 1)\,x + \Big(\frac{\alpha^3}{6} - \frac{\alpha^2}{2}\Big)x^3 + O(x^5) \;\approx\; (\alpha - 1)\,x - \tfrac{1}{3}x^3 \quad (\alpha \approx 1).
$$

This is the normal form of a supercritical pitchfork. For $\alpha < 1$ the balanced state is the single stable equilibrium. At $\alpha = 1$ it loses stability, and two stable herding states branch off with amplitude $x_h \approx \sqrt{3(\alpha - 1)/\alpha^3}$, which equals $\sqrt{3(\alpha - 1)}$ to leading order. The symmetry $x \mapsto -x$ of the $\beta = 0$ model makes optimistic and pessimistic herding equally likely.

| $\alpha$ | 1.01 | 1.05 | 1.1 | 1.3 | 1.5 | 2.0 |
|---|---|---|---|---|---|---|
| exact $x_h$ | 0.1717 | 0.3707 | 0.5029 | 0.7521 | 0.8586 | 0.9575 |
| leading order $\sqrt{3(\alpha-1)/\alpha^3}$ | 0.1706 | 0.3600 | 0.4748 | 0.6400 | 0.6667 | 0.6124 |

The leading-order amplitude is accurate to 1 per cent at $\alpha = 1.01$ and 6 per cent at $\alpha = 1.1$. Beyond $\alpha \approx 1.1$ the exact root is required. Figure `h1_pitchfork.png`.

### 7.2 Folds, cusp, and hysteresis for $\beta \ne 0$

A bias breaks the symmetry and turns the pitchfork into an imperfect bifurcation. Equilibria are created or destroyed in pairs at saddle-node (fold) points, where $F = 0$ and $F' = 0$ hold together:

$$
x_f^2 = 1 - \frac{1}{\alpha}, \qquad \beta_f = \mathrm{artanh}(x_f) - \alpha x_f.
$$

The model is bistable when $|\beta| < \beta_c(\alpha)$ with

$$
\beta_c(\alpha) = \alpha\sqrt{1 - 1/\alpha} - \mathrm{artanh}\sqrt{1 - 1/\alpha} \;\approx\; \tfrac{2}{3}(\alpha - 1)^{3/2} \quad (\alpha \to 1^+).
$$

The two fold curves $\beta = \pm\beta_c(\alpha)$ meet at the cusp $(\alpha, \beta) = (1, 0)$. Figure `h3_cusp.png` shows the bistable wedge.

**Crash mechanism.** Start in the optimistic state with $\beta > 0$ and let news deteriorate, so $\beta$ decreases slowly. The optimistic state persists, and moves little, until $\beta = -\beta_c$, where it merges with the unstable middle state and disappears. The market then falls to the pessimistic state. The size of the fall is set by the fold geometry, independent of the size of the final news change that triggers it.

| $\alpha$ | $\beta_c$ | optimistic state ends at $x$ | falls to $x$ | crash size $\Delta x$ | price fall $\lambda\Delta x$ |
|---|---|---|---|---|---|
| 1.2 | 0.0564 | 0.4082 | −0.7338 | 1.142 | $1.142\lambda$ |
| 1.5 | 0.2075 | 0.5774 | −0.9197 | 1.497 | $1.497\lambda$ |
| 2.0 | 0.5328 | 0.7071 | −0.9868 | 1.694 | $1.694\lambda$ |
| 3.0 | 1.3033 | 0.8165 | −0.9996 | 1.816 | $1.816\lambda$ |

Stronger herding widens the bistable band and enlarges the crash. Recovery requires $\beta$ to rise to $+\beta_c$, so the market returns to optimism at a bias $2\beta_c$ above the bias at which it crashed: 0.415 at $\alpha = 1.5$. Integrating the ODE with $\beta$ falling at $2 \times 10^{-4}$ per day, $x$ crosses 0 at $\beta = -0.2158$, 0.0083 past the fold, which is the delay expected from a slow passage through a saddle-node. Figure `h2_imperfect_hysteresis.png` shows the S-shaped equilibrium curve and the hysteresis loop.

## 8. Finite N: simulation

At finite $N$ the master equation of Section 2 describes the model exactly. The script simulates it with the Gillespie algorithm and computes its stationary law and mean crash times exactly from detailed balance.

### 8.1 Stationary distribution

The chain is one-dimensional, so its stationary law satisfies $\pi(n+1)W_-(n+1) = \pi(n)W_+(n)$. For large $N$, $\pi \propto e^{-N\varphi(x)}$ with $\varphi'(x) = \mathrm{artanh}(x) - \alpha x - \beta$: the minima of $\varphi$ are the stable equilibria of Section 6. The stationary law changes from one peak to two as $\alpha$ crosses a threshold that tends to 1 as $N$ grows:

| $N$ | 25 | 50 | 100 | 200 | 400 | 1000 |
|---|---|---|---|---|---|---|
| bimodal for $\alpha >$ | 0.9634 | 0.9805 | 0.9901 | 0.9950 | 0.9975 | 0.9990 |

The threshold is $1 - 1/N$ to within 0.01 at every $N$ in the table. At $N = 100$ the peaks sit at $x = 0$ ($\alpha = 0.8$), $\pm 0.18$ ($\alpha = 1.0$), $\pm 0.68$ ($\alpha = 1.2$), and $\pm 0.88$ ($\alpha = 1.5$). Figure `h4_finite_N.png` shows the densities and two simulated paths: at $\alpha = 0.8$, $x$ fluctuates about 0; at $\alpha = 1.3$, it stays near one herding state for hundreds of days and then switches to the other.

### 8.2 Spontaneous crashes

With $\beta = 0$ and $\alpha > 1$, finite-$N$ noise carries the market from the optimistic state to the pessimistic state with no change in news. The exact mean time for this passage, from $n_+ = i = N(1 + x_h)/2$ to $n_+ = j = N(1 - x_h)/2$, is the sum of the mean times for each one-step fall:

$$
T = \sum_{k=j+1}^{i} \frac{\sum_{l \ge k}\pi(l)}{\pi(k)\,W_-(k)}.
$$

| $N$ | 20 | 40 | 60 | 80 | 100 | 150 | 200 |
|---|---|---|---|---|---|---|---|
| mean crash time ($\tau$), $\alpha = 1.5$ | 135 | 1,249 | $1.21\times10^4$ | $1.19\times10^5$ | $1.18\times10^6$ | $3.71\times10^8$ | $1.17\times10^{11}$ |

$\ln T$ grows linearly in $N$ with fitted slope 0.1149, against the large-$N$ barrier $\Delta\Phi = \varphi(0) - \varphi(x_h) = 0.1152$, a 0.3 per cent difference. A Monte Carlo check at $N = 30$ (400 Gillespie runs) gives $415 \pm 20$ days against the exact 407.7. Figure `h5_crash_times.png`.

The exponential growth in $N$ constrains calibration. With one unit of $\tau$ equal to one day, a crash every 1.6 to 5 years needs $N$ between 30 and 40. $N$ is then the number of independent decision units in the market (funds, desks, or correlated groups), which is far below the number of individual traders.

## 9. Results for RQ2

1. Herding produces two crash mechanisms. In the deterministic one, a slow deterioration in fundamentals ($\beta$ falling) passes a fold and the market falls by $\Delta x$ = 1.14 to 1.82 without a large shock. In the noise-induced one, a finite population switches between herding states with no change in fundamentals, at a mean interval that grows as $e^{N\Delta\Phi}$.
2. A crash requires $\alpha > 1$. Below this threshold the balanced market is the single stable state and $x$ responds smoothly to $\beta$.
3. Crashes show hysteresis: the fall takes a few units of $\tau$ once the fold is passed, and recovery requires the bias to move back by $2\beta_c$.

## 10. Limitations and next steps

- $\beta$ is exogenous. Lux (1995) adds a trend-following term to $U$, which feeds price changes back into opinion and makes the system two-dimensional. Its stability analysis would use the two-dimensional linear stability of the course notes, by reduction to a second-order ODE.
- Traders are homogeneous and globally coupled. A network makes the threshold depend on network structure.
- $\nu$, $\alpha$, and $\lambda$ are uncalibrated. Matching the crash frequency to the S&P 500 record (34 five-sigma falls in 76.8 years, from the stylised-facts analysis) fixes one combination of $N$, $\alpha$, and $\nu$.
- The volatility model under-produces back-to-back large falls (17 per cent within five days against 42 per cent observed). Switching between herding states is a candidate mechanism for such clusters, and letting the jump intensity of the volatility model depend on $x$ would link the two models.

## References

- Kirman, A. (1993). Ants, rationality, and recruitment. *Quarterly Journal of Economics*, 108(1), 137-156.
- Lux, T. (1995). Herd behaviour, bubbles and crashes. *Economic Journal*, 105(431), 881-896.
- Strogatz, S. H. (2015). *Nonlinear Dynamics and Chaos*, 2nd edition. Westview Press. Chapter 3 (bifurcations, imperfect bifurcations, and hysteresis).
