---
title: "MATH20521 group project: progress report"
subtitle: "Topic 4: Can mathematics explain financial crashes?"
date: "6 November 2026"
geometry: margin=2.2cm
fontsize: 11pt
---

## 1. Project scope

The project asks whether mathematical models can explain financial crashes, using daily S&P 500 data from 1950 to 2026. The work so far has three parts:

1. **Data and benchmark.** Stylised facts of daily returns, a working definition of a large fall, and a test of geometric Brownian motion (GBM) against the data.
2. **Volatility model.** A stochastic-volatility jump-diffusion (Bates) model, written in dimensional and non-dimensional form, with stability analysis, bifurcation analysis, and simulation.
3. **Herding model (RQ2).** A trader-switching model with assumptions, conservation of traders, mean-field ODE, non-dimensionalisation, equilibria, stability, and bifurcation.

## 2. Meetings

| Meeting | Format | Purpose |
|---|---|---|
| 1 | Online | Progress check on the data analysis and the model plan |

## 3. Contributions

| Work | Contributors |
|---|---|
| Data analysis, stylised facts, and GBM comparison | Will M. started the analysis; William Odumosu corrected it (Section 4.2) |
| Model equations and non-dimensionalisation (volatility model) | Will M. and William Odumosu |
| Stability analysis, bifurcation analysis, and simulation (volatility model) | William Odumosu |
| Herding model: stability and bifurcation analysis (RQ2) | William Odumosu |

## 4. Work completed

### 4.1 Data and the definition of a large fall

The data are 19,319 daily log returns of the S&P 500 (^SPX, via yfinance), 4 January 1950 to 16 October 2026. The daily standard deviation is 0.995 per cent (15.8 per cent a year). Returns have excess kurtosis 25.2, skewness −0.95, a loss-tail index of 2.8 to 3.1, and volatility clustering, with autocorrelation of $|r_t|$ of 0.27 at lag 1 and 0.05 at lag 250.

We define a fall as large when the standardised return $z_t = (r_t - \hat m)/\hat\sigma$ is below $-3$. GBM's error grows from 9 per cent at $z = -2$ to a factor of 5.9 at $z = -3$, so $-3$ marks where GBM's error becomes large. The sample has 155 large falls (2.0 a year) and 34 extreme falls below $-5$.

### 4.2 GBM comparison and corrections

GBM expects 0.0055 falls below $-5\sigma$ in the sample; 34 occurred, a ratio of about 6,100. GBM expects one 5 per cent fall every 37,000 years; one occurred every 2.6 years. Measuring σ from the previous 250 days lowers the count to 26, which is still 4,800 times the GBM expectation. 42 per cent of 3 per cent falls are followed by another within five trading days, against 3.4 per cent under GBM.

The first version of the analysis contained four errors, which were corrected:

| Error in first version | Effect | Correction |
|---|---|---|
| The count of 30 compared with GBM's 5σ expectation was the number of falls of 5 per cent or more | Ratio 30 : 0.0055 compared different thresholds | Like-for-like ratios: 30 : 0.0021 for 5 per cent falls; 34 : 0.0055 for 5σ falls |
| Rolling σ written `=STDEV.S(M5,M254)` | Standard deviation of two cells; 1,109 "5σ falls" | `=STDEV.S(M5:M254)`; 26 falls |
| Daily σ taken as 1.1 per cent | 3 per cent fall stated as 2.8σ, expected once every 1.5 years | Sample σ 0.995 per cent: 3.09σ, expected once every 4.0 years; 132 occurred |
| Number of returns taken as `=COUNT(B:B)` = 19,320 | Counted closing prices | 19,319 returns |

### 4.3 Volatility model

The Bates model combines stochastic variance (Heston) with jumps (Merton). Non-dimensionalising with the trading day as the time scale gives eight dimensionless groups. Three are small: one day's long-run σ ($s \approx 0.01$), the daily Sharpe ratio ($m \approx 0.03$), and the variance reversion rate per day ($\varepsilon \approx 0.011$, a half-life of 62 days). The small value of $\varepsilon$ means the variance is effectively fixed over one day.

### 4.4 Volatility model: stability, bifurcation, and simulation

- **Stability.** Without noise or jumps, the model's equilibria form a line on which the variance is at its long-run value, with eigenvalues $0$ and $-\varepsilon$. The variance is globally stable and the price level is neutrally stable. All moments of the variance converge.
- **Linearisation.** At the estimated parameter values, linearising the variance equation assigns 9 to 16 per cent probability to negative variance, so the analysis uses the exact distribution.
- **Bifurcations.** The stationary variance distribution changes shape at $\delta = 1$, the Feller threshold. The one-day return distribution gains a second peak, a distinct crash mode, when the mean jump size is below a critical value $a^*$. $a^*$ is −5.02 without stochastic variance and −7.56 at $\delta = 1$.
- **Simulation.** We ran 200 simulated histories of the sample's length for GBM, Merton, Heston, and Bates. Bates is the one model whose 90 per cent bands contain the observed counts of 3σ falls (155), 5σ falls (34), and 5 per cent falls (30). All four models under-produce the clustering of large falls: Bates gives 17 per cent within five days against 42 per cent observed.

### 4.5 Herding model (RQ2)

$N$ traders switch between optimism and pessimism. The incentive $U = \alpha x + \beta$ depends on the opinion index $x = (n_+ - n_-)/N$, the herding strength $\alpha$, and a news bias $\beta$. Conservation of traders, $n_+ + n_- = N$, reduces the model to one variable. Non-dimensionalising time by $1/(2\nu)$, where $\nu$ is the rate at which each trader reconsiders, gives the mean-field ODE

$$
\frac{dx}{d\tau} = \cosh(\alpha x + \beta)\big[\tanh(\alpha x + \beta) - x\big].
$$

- **Equilibria and stability.** Equilibria satisfy $x^* = \tanh(\alpha x^* + \beta)$. An equilibrium is stable when $\alpha(1 - x^{*2}) < 1$.
- **Pitchfork bifurcation.** At $\alpha = 1$ with $\beta = 0$, the balanced market loses stability and two stable herding states appear (±0.8586 at $\alpha = 1.5$).
- **Folds and hysteresis.** For $\beta \ne 0$ the pitchfork becomes two folds meeting at a cusp at $(\alpha, \beta) = (1, 0)$. When the bias falls past $-\beta_c$, the optimistic state disappears and the market falls to the pessimistic state. At $\alpha = 1.5$, $\beta_c = 0.2075$ and the fall in $x$ is 1.50.
- **Finite-N simulation.** Finite-$N$ noise produces crashes with no change in news. The mean time between them grows as $e^{0.115N}$, in agreement with the theoretical barrier of 0.1152. A crash every 1.6 to 5 years requires $N$ between 30 and 40, so $N$ represents independent decision units such as funds, not individual traders.

## 5. Next steps

| Task | Detail |
|---|---|
| Calibration | Fit $\alpha$, $\nu$, and $N$ in the herding model to the observed crash frequency |
| Two-dimensional herding model | Add trend-following (Lux, 1995) and analyse its stability with the course's 2-d method |
| Link the two models | Let the jump rate of the volatility model depend on the herding state $x$, to test whether this reproduces the clustering of large falls |
| Poster and presentation | Select figures and results; submission 7 December (date to be confirmed on Canvas); presentations Friday 11 December |
| Task allocation | Assign the remaining tasks at the next group meeting |

## 6. Files

| File | Contents |
|---|---|
| `sp500_large_falls.py` | Data analysis, stylised facts, GBM comparison, and corrected spreadsheet calculations |
| `crash_model.py` | Volatility model: stability, bifurcation, and simulation |
| `stability_bifurcation_simulation.md` | Write-up of the volatility-model analysis |
| `herding_model.py` | Herding model: equilibria, stability, bifurcation, and finite-N simulation |
| `herding_model.md` | Write-up of the herding model (RQ2) |
