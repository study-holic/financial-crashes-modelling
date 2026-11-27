# Can mathematics explain financial crashes?

This repository holds the code and write-ups for our MATH20521 Principles of Mathematical Modelling group project (University of Manchester, 2026/27, topic 4). We test geometric Brownian motion (GBM) against 76.8 years of daily S&P 500 returns, fit a stochastic-volatility jump model that reproduces the observed frequency of large falls, and analyse a herding model in which crashes come from traders copying one another.

## Results

- GBM expects 0.0055 daily falls below −5σ between January 1950 and October 2026. The index had 34.
- In 200 simulated histories of the Bates model, the 90 per cent bands contain the observed counts of 3σ falls (155), 5σ falls (34), and 5 per cent falls (30). GBM, Merton, and Heston each miss at least one of the three.
- Every model we simulated under-produces clustering. In the data, 42 per cent of 3 per cent falls are followed by another within five trading days; the Bates simulations give 17 per cent.
- In the herding model, the balanced market loses stability at herding strength α = 1. Above that value, a slow decline in sentiment carries the market past a fold, and the opinion index drops by 1.14 to 1.82 as α runs from 1.2 to 3.
- With no change in news, a finite population of N traders still switches from optimism to pessimism. At α = 1.5 the mean time between these crashes grows as e^(0.115N).

## Repository layout

| Path | Contents |
|---|---|
| `src/sp500_large_falls.py` | Stylised facts, the definition of a large fall, the GBM comparison, the corrected spreadsheet calculations, and screening values for the Bates model's dimensionless groups |
| `src/crash_model.py` | Bates model: stability of the noise-free system, moment stability, two bifurcations, and Monte Carlo simulation of GBM, Merton, Heston, and Bates |
| `src/herding_model.py` | Herding model: equilibria, linear stability, pitchfork and fold bifurcations, hysteresis, and finite-N Gillespie simulation |
| `docs/stability_bifurcation_simulation.md` | Write-up of the Bates model analysis |
| `docs/herding_model.md` | Write-up of the herding model (RQ2) |
| `docs/progress_report.md` | Progress report of 6 November 2026 |
| `data/sp500_1950_2026.xlsx` | Daily S&P 500 (^SPX) prices from Yahoo Finance, downloaded with yfinance, 3 January 1950 to 16 October 2026 |
| `figures/` | The 17 figures written by the three scripts |

## Running the code

We tested the scripts with Python 3.13. Run them from the repository root, so that each one writes its figures to `figures/`.

```bash
pip install -r requirements.txt
python src/sp500_large_falls.py data/sp500_1950_2026.xlsx   # about 10 s
python src/crash_model.py data/sp500_1950_2026.xlsx         # about 1 min
python src/herding_model.py                                 # about 10 s
```

`crash_model.py` also runs without the data file and then uses S&P 500 statistics stored in the script. The simulations use the fixed seed 20261106, so a rerun reproduces every number in `docs/`.

| Figures | Script | Subject |
|---|---|---|
| `fig1` to `fig7` | `sp500_large_falls.py` | Returns and volatility, autocorrelation, leverage, kurtosis by horizon, yearly 3 per cent falls, loss tail, observed against GBM counts |
| `s1` to `s5` | `crash_model.py` | Phase portrait, variance-density bifurcation, crash-mode bifurcation, simulated paths, simulations against the data |
| `h1` to `h5` | `herding_model.py` | Pitchfork, hysteresis loop, cusp region, finite-N paths and distributions, mean crash time against N |

## Models

The Bates model, in units of one trading day and one day's long-run standard deviation, is

$$
dX = \Big(m - \tfrac{s}{2}V\Big)\,d\tau + \sqrt{V}\,dB^{(1)} + Z\,dN, \qquad
dV = \varepsilon(1 - V)\,d\tau + \sqrt{2\varepsilon\delta V}\,dB^{(2)},
$$

with $dB^{(1)}dB^{(2)} = \rho\,d\tau$, jump probability $\Lambda$ per day, and jump sizes $Z \sim N(a, b^2)$. Setting $\delta = \Lambda = 0$ recovers GBM.

The herding model tracks the opinion index $x = (n_+ - n_-)/N$ of $N$ traders. In time units of $1/(2\nu)$ its mean-field equation is

$$
\frac{dx}{d\tau} = \cosh(\alpha x + \beta)\big[\tanh(\alpha x + \beta) - x\big],
$$

where $\alpha$ is the herding strength and $\beta$ is the bias from news. The two write-ups in `docs/` derive both models and give the stability and bifurcation results in full.

## The data file

`data/sp500_1950_2026.xlsx` keeps the formulas of our first spreadsheet analysis, including the rolling-σ formula `=STDEV.S(M5,M254)` in column N. The comma makes that formula the standard deviation of two cells, which gave 1,109 "5σ falls". `sp500_large_falls.py` reproduces the error, giving 1,110 (the sheet's range ends one row earlier), and computes the corrected count, 26, from `STDEV.S(M5:M254)`. Section 4.2 of `docs/progress_report.md` lists all four corrections.

## Contributors

Will M. started the data analysis and the GBM comparison, and William Odumosu corrected them. The two wrote the model equations and their non-dimensionalisation together. William Odumosu did the stability, bifurcation, and simulation work for both models.

## References

- Andersen, L. (2008). Simple and efficient simulation of the Heston stochastic volatility model. *Journal of Computational Finance*, 11(3), 1-42.
- Bates, D. S. (1996). Jumps and stochastic volatility: exchange rate processes implicit in Deutsche Mark options. *Review of Financial Studies*, 9(1), 69-107.
- Cont, R. (2001). Empirical properties of asset returns: stylized facts and statistical issues. *Quantitative Finance*, 1(2), 223-236.
- Heston, S. L. (1993). A closed-form solution for options with stochastic volatility with applications to bond and currency options. *Review of Financial Studies*, 6(2), 327-343.
- Kirman, A. (1993). Ants, rationality, and recruitment. *Quarterly Journal of Economics*, 108(1), 137-156.
- Lux, T. (1995). Herd behaviour, bubbles and crashes. *Economic Journal*, 105(431), 881-896.
- Merton, R. C. (1976). Option pricing when underlying stock returns are discontinuous. *Journal of Financial Economics*, 3(1-2), 125-144.
