"""
S&P 500 large falls: stylised facts, GBM comparison, and model screening values.

Usage:
    python src/sp500_large_falls.py data/sp500_1950_2026.xlsx
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import acf
from statsmodels.stats.diagnostic import acorr_ljungbox
from arch import arch_model

FIG = Path("figures"); FIG.mkdir(exist_ok=True)   # figures go here

KS = [2, 3, 4, 5, 6, 7, 10]                            # thresholds in sigmas
PCTS = [0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.20]      # thresholds as percentage falls


# 1. Data

def load(path):
    """Closes and volumes from the yfinance spreadsheet, indexed by date."""
    raw = pd.read_excel(path, header=None, skiprows=3, usecols=[0, 1, 5])   # Date, Close, Volume
    raw.columns = ["date", "close", "volume"]
    df = raw.dropna(subset=["date", "close"]).copy()                         # drop empty rows
    df["date"] = pd.to_datetime(df["date"])
    df["close"] = df["close"].astype(float)
    df["volume"] = pd.to_numeric(df["volume"], errors="coerce")             # bad cells become NaN
    return df.set_index("date").sort_index()


def hill(x_desc, k):
    """Hill tail index from the k largest values (sorted largest first); about 3 means a fat tail."""
    return 1.0 / np.mean(np.log(x_desc[:k] / x_desc[k]))


# 2. Stylised facts

def stylised_facts(df, r):
    """Standard statistical patterns of daily returns r."""
    M = r.values                  # returns as a NumPy array
    N = len(M)
    out = {}                      # every result goes in here
    out["N"] = N
    out["years"] = (r.index[-1] - df.index[0]).days / 365.25

    # Summary numbers
    out["mean"], out["sd"] = M.mean(), M.std(ddof=1)                  # ddof=1: sample sd
    out["ann_vol"] = out["sd"] * np.sqrt(252)                         # daily -> yearly
    out["skew"], out["exkurt"] = stats.skew(M), stats.kurtosis(M)     # normal: both 0
    rx = r[r.index != "1987-10-19"]                                   # without the 1987 crash
    out["skew_ex1987"], out["exkurt_ex1987"] = stats.skew(rx), stats.kurtosis(rx)
    out["JB"] = stats.jarque_bera(M).statistic                        # normality test; > 5.99 rejects
    out["largest_fall"] = (r.idxmin().date(), np.expm1(r.min()))      # expm1: log return -> % change
    out["largest_rise"] = (r.idxmax().date(), np.expm1(r.max()))

    # Tails
    L = np.sort(-M[M < 0])[::-1]                                      # loss sizes, largest first
    G = np.sort(M[M > 0])[::-1]                                       # gain sizes, largest first
    out["hill_loss"] = {k: hill(L, k) for k in (50, 100, 200, 400)}
    out["hill_gain"] = {k: hill(G, k) for k in (50, 100, 200, 400)}
    out["t_fit"] = stats.t.fit(M)                                     # (df, centre, scale); low df = fat tails

    # Autocorrelation
    out["acf_r"] = acf(M, nlags=250, fft=True)[1:]                    # returns: near zero
    out["acf_abs"] = acf(np.abs(M), nlags=250, fft=True)[1:]          # |r|: positive = clustering
    out["band"] = 1.96 / np.sqrt(N)                                   # 95% band around zero
    out["LB10_r"] = acorr_ljungbox(M, lags=[10]).iloc[0, 0]           # joint test of lags 1-10
    out["LB10_abs"] = acorr_ljungbox(np.abs(M), lags=[10]).iloc[0, 0]

    # Decay of the |r| autocorrelation: power law vs exponential
    lags = np.arange(1, 251)
    la = np.log(out["acf_abs"])
    bp, cp = np.polyfit(np.log(lags), la, 1)                          # log acf = c - beta log(lag)
    be, ce = np.polyfit(lags, la, 1)                                  # log acf = c - rate * lag
    out["powerlaw_beta"] = -bp
    out["rss_power"] = np.sum((la - (bp * np.log(lags) + cp)) ** 2)   # smaller RSS fits better
    out["rss_exp"] = np.sum((la - (be * lags + ce)) ** 2)

    # Leverage: today's return against the size of the return k days later
    out["lev"] = np.array([np.corrcoef(M[:-k], np.abs(M[k:]))[0, 1] for k in range(1, 51)])
    out["lev_rev"] = np.array([np.corrcoef(np.abs(M[:-k]), M[k:])[0, 1] for k in range(1, 51)])   # reverse: ~0

    # Aggregational Gaussianity: longer horizons look more normal
    lp = np.log(df["close"])
    out["agg"] = {}
    for h in (1, 2, 5, 10, 21, 63, 126, 252):
        rh = lp.diff(h).dropna().iloc[::h]                            # non-overlapping h-day returns
        out["agg"][h] = (stats.kurtosis(rh), stats.skew(rh), len(rh))

    # Clustering of 3% falls
    pc = np.expm1(r)                                                  # daily % change
    ev = np.where(pc.values <= -0.03)[0]                              # days with a 3%+ fall
    w = np.diff(ev)                                                   # waits between them
    p = len(ev) / N                                                   # daily chance of one
    out["wait"] = dict(n=len(ev), mean=w.mean(), median=np.median(w), cv=w.std() / w.mean(),
                       within5=np.mean(w <= 5), within5_indep=1 - (1 - p) ** 5)   # GBM: CV ~ 1
    yearly = (pc <= -0.03).groupby(pc.index.year).sum()               # 3% falls per year
    out["yearly3"] = yearly
    out["dispersion"] = yearly.var() / yearly.mean()                  # Poisson gives 1

    # Volume and volatility
    dv = np.log(df["volume"].replace(0, np.nan)).diff()               # change in log volume
    j = pd.concat([r.abs(), dv], axis=1).dropna()                     # align by date
    out["corr_vol_volume"] = np.corrcoef(j.iloc[:, 0], j.iloc[:, 1])[0, 1]

    out["decade_vol"] = r.groupby((r.index.year // 10) * 10).std() * np.sqrt(252)   # 1987 -> 1980s
    return out


# 3. Standardisations

def standardise(r):
    """Sigma moves z = r / sigma under three definitions of sigma."""
    mu, sd = r.mean(), r.std(ddof=1)
    z = {"unconditional": (r - mu) / sd}                              # one sigma, as in GBM

    roll = r.rolling(250).std().shift(1)                              # previous 250 days, excluding today
    z["rolling250"] = (r / roll).dropna()                             # corrected STDEV.S(M5:M254)

    # GARCH: volatility updated daily from past returns (fitted in per cent)
    gjr = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, o=1, q=1, dist="t").fit(disp="off")   # falls raise vol more
    garch = arch_model(r * 100, mean="Constant", vol="GARCH", p=1, q=1, dist="normal").fit(disp="off")
    z["gjr_t"] = gjr.std_resid.dropna()                               # return / GARCH sigma
    z["garch_normal"] = garch.std_resid.dropna()
    return z, gjr, garch


def spreadsheet_bug_count(r):
    """Count of '5-sigma falls' under the original =STDEV.S(M5,M254) (sd of two cells)."""
    M = r.values
    sig = np.full(len(M), np.nan)
    for i in range(251, len(M)):
        sig[i] = np.std([M[i - 250], M[i - 1]], ddof=1)   # row i+4 uses M(i+4-250), M(i+3)
    with np.errstate(divide="ignore", invalid="ignore"):  # hide divide-by-zero warnings
        return int(np.nansum(M / sig < -5))


# 4. GBM comparison

def gbm_sigma_table(z):
    """Observed days with z < -k against GBM's expected N * Phi(-k)."""
    rows = []
    for k in KS:
        row = {"k": k}
        for name, s in z.items():
            n = len(s)
            obs = int((s < -k).sum())                     # observed
            exp = n * stats.norm.cdf(-k)                  # GBM expected
            row[f"{name}_obs"] = obs
            row[f"{name}_exp"] = exp
            row[f"{name}_ratio"] = obs / exp
            row[f"{name}_pois_p"] = stats.poisson.sf(obs - 1, exp) if obs else 1.0   # P(>= obs) under GBM
        rows.append(row)
    return pd.DataFrame(rows).set_index("k")


def gbm_pct_table(r, years, t_params, gpd):
    """Same comparison for fixed percentage falls, with Student-t and GPD expectations."""
    M = r.values; N = len(M)
    mu, sd = M.mean(), M.std(ddof=1)
    pc = np.expm1(M)
    u, xi, beta, n_exc = gpd
    rows = []
    for th in PCTS:
        x = np.log(1 - th)                                # fall as a log return
        p = stats.norm.cdf((x - mu) / sd)                 # GBM daily probability
        obs = int((pc <= -th).sum())
        loss = -x
        gpd_exp = N * (n_exc / N) * stats.genpareto.sf(loss - u, xi, 0, beta) if loss > u else np.nan   # valid above u
        rows.append(dict(fall=th, sigma_move=(x - mu) / sd, gbm_exp=N * p,
                         gbm_years_per_event=years / (N * p), observed=obs,
                         obs_per_year=obs / years, ratio=obs / (N * p),
                         student_t_exp=N * stats.t.cdf(x, *t_params), gpd_exp=gpd_exp))
    return pd.DataFrame(rows).set_index("fall")


def fit_gpd(r, q=0.95):
    """Generalised Pareto fit to the largest 5% of losses: (u, xi, beta, n)."""
    loss = -r.values
    u = np.quantile(loss, q)                              # 95th-percentile loss
    exc = loss[loss > u] - u                              # excess over u
    xi, _, beta = stats.genpareto.fit(exc, floc=0)        # xi > 0: power-law tail
    return u, xi, beta, len(exc)


# 5. Model screening values

def screening_groups(r, garch, gjr, facts):
    """Order-of-magnitude values of the Bates model's dimensionless groups (not a calibration)."""
    a1, b1 = garch.params["alpha[1]"], garch.params["beta[1]"]
    h = (garch.conditional_volatility / 100) ** 2         # daily variance
    zg = garch.std_resid.dropna()
    jumps = zg[zg < -4]                                   # jump candidates
    hg = (gjr.conditional_volatility / 100) ** 2
    eps = 1 - (a1 + b1)                                   # variance reversion per day
    kurt252 = facts["agg"][252][0]
    return dict(
        s=np.sqrt(h.mean()),                               # one day's long-run sigma
        m=r.mean() / r.std(ddof=1),                        # daily Sharpe ratio
        eps=eps,
        half_life_days=np.log(0.5) / np.log(a1 + b1),      # days for a vol shock to halve
        delta_from_garch=h.var() / h.mean() ** 2,          # variance of variance
        delta_from_daily_kurt=facts["exkurt_ex1987"] / 3,  # exkurt = 3 delta
        delta_from_252d_kurt=kurt252 * eps * 252 / 6,      # exkurt_h = 6 delta / (eps h)
        rho=np.corrcoef(r.values[:-1], np.diff(hg.values))[0, 1],   # return vs next variance change
        Lambda=len(jumps) / len(r),                        # jump probability per day
        a=jumps.mean(), b=jumps.std(),                     # jump mean and spread, in sigmas
    )


def frozen_variance_counts(N, delta, Lam, a, b, ks=(2, 3, 5, 10), draws=4_000_000, seed=1):
    """Model's expected falls beyond k sigma, with V fixed within a day and Gamma across days."""
    if delta == 0:
        V = np.ones(1)                                     # constant variance (GBM)
    else:
        V = stats.gamma(1 / delta, scale=delta).rvs(draws, random_state=np.random.default_rng(seed))
    res = {}
    for k in ks:
        p = (1 - Lam) * stats.norm.cdf(-k / np.sqrt(V))                    # no-jump days
        if Lam:
            p = p + Lam * stats.norm.cdf((-k - a) / np.sqrt(V + b * b))    # jump days
        res[k] = N * p.mean()                              # expected count
    return res


# 6. Figures

def figures(r, gjr, facts, sig_tab, gpd):
    """Save figures fig1 to fig7."""
    M = r.values; N = len(M); mu, sd = M.mean(), M.std(ddof=1)

    # Returns and GARCH volatility
    fig, ax = plt.subplots(2, 1, figsize=(11, 6), sharex=True)
    ax[0].plot(r.index, M * 100, lw=0.4); ax[0].set_ylabel("log return, %")
    ax[1].plot(r.index, gjr.conditional_volatility * np.sqrt(252), lw=0.7, color="C1")
    ax[1].set_ylabel("annualised σ, %"); fig.tight_layout(); fig.savefig(FIG / "fig1_returns_vol.png", dpi=150)

    # Autocorrelation with 95% band
    fig, ax = plt.subplots(figsize=(7, 4))
    lags = np.arange(1, 251)
    ax.plot(lags, facts["acf_abs"], label="|r|"); ax.plot(lags, facts["acf_r"], lw=0.8, label="r")
    ax.axhline(facts["band"], ls="--", c="grey"); ax.axhline(-facts["band"], ls="--", c="grey")
    ax.set_xlabel("lag, trading days"); ax.set_ylabel("autocorrelation"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig2_acf.png", dpi=150)

    # Leverage
    fig, ax = plt.subplots(figsize=(7, 4))
    k = np.arange(1, 51)
    ax.plot(k, facts["lev"], label="corr(r_t, |r_t+k|)"); ax.plot(k, facts["lev_rev"], label="corr(|r_t|, r_t+k)")
    ax.axhline(0, c="grey", lw=0.8); ax.set_xlabel("k"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig3_leverage.png", dpi=150)

    # Kurtosis against horizon
    fig, ax = plt.subplots(figsize=(7, 4))
    h = list(facts["agg"]); ax.plot(h, [facts["agg"][x][0] for x in h], "o-")
    ax.set_xscale("log"); ax.set_xlabel("horizon h, days"); ax.set_ylabel("excess kurtosis")
    fig.tight_layout(); fig.savefig(FIG / "fig4_aggregation.png", dpi=150)

    # 3% falls per year
    fig, ax = plt.subplots(figsize=(9, 4))
    y = facts["yearly3"]; ax.bar(y.index, y.values)
    ax.set_ylabel("days with fall ≥ 3%"); fig.tight_layout(); fig.savefig(FIG / "fig5_yearly.png", dpi=150)

    # Loss tail on log-log axes
    fig, ax = plt.subplots(figsize=(7, 5))
    L = np.sort(-M[M < 0])[::-1]
    ax.loglog(L * 100, np.arange(1, len(L) + 1) / N, ".", ms=3, label="empirical")   # i-th largest exceeded on i days
    xs = np.logspace(np.log10(0.003), np.log10(0.26), 200)                            # 0.3% to 26%
    ax.loglog(xs * 100, stats.norm.cdf((-xs - mu) / sd), label="Gaussian (GBM)")
    ax.loglog(xs * 100, stats.t.cdf(-xs, *facts["t_fit"]), label="Student-t")
    u, xi, beta, n_exc = gpd; m = xs > u                                              # GPD above u
    ax.loglog(xs[m] * 100, (n_exc / N) * stats.genpareto.sf(xs[m] - u, xi, 0, beta), label="GPD")
    ax.set_ylim(3e-5, 1); ax.set_xlabel("daily log loss x, %"); ax.set_ylabel("P(loss > x)"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig6_tail.png", dpi=150)

    # Observed / GBM expected
    fig, ax = plt.subplots(figsize=(7, 4))
    for name in ("unconditional", "rolling250", "gjr_t"):
        ax.semilogy(sig_tab.index, sig_tab[f"{name}_ratio"], "o-", label=name)
    ax.set_xlabel("threshold k (σ)"); ax.set_ylabel("observed ÷ GBM expected"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "fig7_ratio.png", dpi=150)
    plt.close("all")                                      # free memory


# Main

def main(path):
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 30)   # wide tables

    df = load(path)
    r = np.log(df["close"]).diff().dropna()               # r_t = ln S_t - ln S_(t-1)

    # Stylised facts
    facts = stylised_facts(df, r)
    print(f"N = {facts['N']}, {facts['years']:.1f} years, "
          f"mean {facts['mean']*100:.4f}%/day, sd {facts['sd']*100:.3f}%/day ({facts['ann_vol']*100:.1f}% a year)")
    print(f"skew {facts['skew']:.2f} (ex-1987 {facts['skew_ex1987']:.2f}), "
          f"excess kurtosis {facts['exkurt']:.1f} (ex-1987 {facts['exkurt_ex1987']:.1f}), JB {facts['JB']:.0f}")
    print("largest fall", facts["largest_fall"], "largest rise", facts["largest_rise"])
    print("Hill losses", {k: round(v, 2) for k, v in facts["hill_loss"].items()},
          "gains", {k: round(v, 2) for k, v in facts["hill_gain"].items()})
    print("Student-t df", round(facts["t_fit"][0], 2))
    print("ACF r lags 1-5", np.round(facts["acf_r"][:5], 3), "band", round(facts["band"], 3))
    print("ACF |r| lags 1,10,100,250", np.round(facts["acf_abs"][[0, 9, 99, 249]], 3))
    print(f"Ljung-Box Q(10): r {facts['LB10_r']:.1f}, |r| {facts['LB10_abs']:.0f}")
    print(f"power-law beta {facts['powerlaw_beta']:.2f}; RSS power {facts['rss_power']:.2f} vs exp {facts['rss_exp']:.2f}")
    print("leverage k=1,2,5", np.round(facts["lev"][[0, 1, 4]], 3))
    print("aggregation (exkurt, skew, n):", {h: tuple(round(x, 2) for x in v) for h, v in facts["agg"].items()})
    print("3% falls waiting times", {k: round(v, 3) for k, v in facts["wait"].items()})
    print(f"yearly dispersion index {facts['dispersion']:.1f}; years with none {(facts['yearly3'] == 0).sum()}")
    print(f"corr(|r|, dlog volume) {facts['corr_vol_volume']:.2f}")
    print("annualised vol by decade\n", (facts["decade_vol"] * 100).round(1))

    # Spreadsheet check
    print(f"\nSpreadsheet bug STDEV.S(M5,M254): {spreadsheet_bug_count(r)} 'five-sigma falls'")

    # Sigma moves and GARCH fits
    z, gjr, garch = standardise(r)
    print("\nGJR-GARCH-t params\n", gjr.params.round(4))
    print("GARCH(1,1) params\n", garch.params.round(4))
    for name, s in z.items():
        print(f"excess kurtosis of {name} z: {stats.kurtosis(s):.2f}")

    # Tail fit
    gpd = fit_gpd(r)
    print(f"\nGPD above {gpd[0]*100:.2f}% loss: xi {gpd[1]:.3f}, beta {gpd[2]*100:.3f}%, n {gpd[3]}")

    # GBM comparison (p-value columns left out of the print)
    sig_tab = gbm_sigma_table(z)
    print("\nFalls in sigma units vs GBM\n",
          sig_tab[[c for c in sig_tab.columns if not c.endswith("pois_p")]].to_string(float_format=lambda v: f"{v:.3g}"))
    pct_tab = gbm_pct_table(r, facts["years"], facts["t_fit"], gpd)
    print("\nFixed percentage falls vs GBM\n", pct_tab.to_string(float_format=lambda v: f"{v:.3g}"))

    # Screening values
    g = screening_groups(r, garch, gjr, facts)
    print("\nDimensionless groups (screening magnitudes)")
    for k, v in g.items():
        print(f"  {k:24s} {v: .4g}")

    # Predicted large-fall counts: (label, delta, Lambda); 0 switches a part off
    N = len(r)
    print("\nFrozen-variance expected counts, k = 2, 3, 5, 10")
    specs = [("GBM", 0, 0), ("Heston d=1", 1, 0), ("Bates d=1", 1, g["Lambda"]), ("Heston d=4", 4, 0)]
    for label, d, Lam in specs:
        c = frozen_variance_counts(N, d, Lam, g["a"], g["b"])
        print(f"  {label:12s}", "  ".join(f"{v:9.3g}" for v in c.values()))
    print("  observed    ", "  ".join(f"{int((z['unconditional'] < -k).sum()):9d}" for k in (2, 3, 5, 10)))

    figures(r, gjr, facts, sig_tab, gpd)
    print(f"\nFigures saved to {FIG.resolve()}")


if __name__ == "__main__":
    main(sys.argv[1])                                     # spreadsheet path from the command line
