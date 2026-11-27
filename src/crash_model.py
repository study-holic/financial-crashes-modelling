"""
Bates crash model: stability, bifurcation, and simulation.

Usage:
    pip install numpy pandas scipy matplotlib openpyxl
    python src/crash_model.py                                  # stored S&P 500 statistics
    python src/crash_model.py data/sp500_1950_2026.xlsx       # recompute them from the data
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats, special

FIG = Path("figures"); FIG.mkdir(exist_ok=True)   # figures go here
RNG_SEED = 20261106                               # fixed seed so runs repeat

# Dimensionless groups (screening values from the data)
BASE = dict(
    s=0.0100,     # one day's long-run sigma
    m=0.032,      # daily Sharpe ratio
    eps=0.0111,   # variance reversion rate per day
    delta=1.0,    # variance of V (Feller threshold at 1)
    rho=-0.43,    # price-variance shock correlation
    Lam=0.00197,  # jump probability per day
    a=-5.42,      # mean jump, in daily sigmas
    b=1.84,       # jump spread, in daily sigmas
)

# Nested models: each switches part of Bates off
SPECS = {
    "GBM":    dict(delta=0.0, Lam=0.0),
    "Merton": dict(delta=0.0),
    "Heston": dict(Lam=0.0),
    "Bates":  dict(),
}


def params(name):
    """Full parameter set for one named model."""
    p = dict(BASE); p.update(SPECS[name]); return p


# 1. Deterministic skeleton

def skeleton_analysis(p):
    """Linear stability of the noise-free system in (X~, u = V - 1)."""
    J = np.array([[0.0, -p["s"] / 2], [0.0, -p["eps"]]])     # Jacobian
    eig, vec = np.linalg.eig(J)                               # eigenvalues 0 and -eps
    out = dict(J=J, eigenvalues=eig, eigenvectors=vec)        # collect results
    out["permanent_shift_per_unit_u0"] = -p["s"] / (2 * p["eps"])   # price drag from a variance shock
    out["half_life_days"] = np.log(2) / p["eps"]
    out["char_roots"] = np.roots([1.0, p["eps"], 0.0])             # roots of r(r + eps) = 0
    return out


# 2. Moment stability and linearisation

def moment_system(p, n_max=4):
    """Linear ODEs for M_n = E[V^n]; eigenvalues are -n*eps."""
    eps, d = p["eps"], p["delta"]
    A = np.zeros((n_max, n_max)); c = np.zeros(n_max)            # dM/dτ = A M + c
    for n in range(1, n_max + 1):
        A[n - 1, n - 1] = -n * eps                              # decay of M_n
        if n == 1:
            c[0] = eps                                          # M_0 = 1 enters as a constant
        else:
            A[n - 1, n - 2] = n * eps * (1 + (n - 1) * d)       # feed from M_(n-1)
    M_stat = np.linalg.solve(A, -c)                             # steady state: dM/dτ = 0
    gamma_moments = [np.prod([1 + j * d for j in range(n)]) for n in range(1, n_max + 1)]
    return dict(A=A, eigenvalues=np.diag(A), stationary=M_stat, gamma_check=np.array(gamma_moments))


def linearisation_check(deltas=(0.05, 0.25, 0.57, 1.0, 2.0, 4.0)):
    """Compare the linearised law N(1, δ) with the exact Gamma(1/δ, δ)."""
    rows = []
    for d in deltas:
        g = stats.gamma(1 / d, scale=d)          # exact
        n = stats.norm(1, np.sqrt(d))            # linearised
        rows.append(dict(delta=d,
                         P_negative_linear=n.cdf(0),          # impossible negative variance
                         P_V_gt_3_exact=g.sf(3), P_V_gt_3_linear=n.sf(3),
                         P_V_lt_0p1_exact=g.cdf(0.1), P_V_lt_0p1_linear=n.cdf(0.1) - n.cdf(0)))
    return pd.DataFrame(rows).set_index("delta")


# 3a. Variance-density bifurcation at δ = 1

def variance_density(v, delta):
    """Stationary density of V: Gamma(1/δ, δ)."""
    return stats.gamma(1 / delta, scale=delta).pdf(v)


def variance_mode(delta):
    """Most likely V: 1 - δ below the threshold, 0 above."""
    return max(1 - delta, 0.0)


# 3b. Crash-mode bifurcation in the one-day return density

def one_day_density(x, delta, Lam, a, b, n_quad=120):
    """One-day density with V frozen: normal mixture plus a jump component."""
    x = np.asarray(x, float)                                    # points to evaluate at
    if delta == 0:
        V = np.array([1.0]); w = np.array([1.0])                # constant variance
    else:
        alpha = 1 / delta
        u, w = special.roots_genlaguerre(n_quad, alpha - 1)     # quadrature over the Gamma law
        V = delta * u; w = w / special.gamma(alpha)
    X = x[:, None]                                              # broadcast x against the V nodes
    core = (w * stats.norm.pdf(X, 0, np.sqrt(V))).sum(1)                 # ordinary days
    jump = (w * stats.norm.pdf(X, a, np.sqrt(V + b * b))).sum(1)         # jump days
    return (1 - Lam) * core + Lam * jump


def n_modes(delta, Lam, a, b, grid=np.linspace(-45, -0.5, 6000)):
    """Number of peaks: the central one plus any on the loss side."""
    f = one_day_density(grid, delta, Lam, a, b)
    interior = (f[1:-1] > f[:-2]) & (f[1:-1] > f[2:])          # local maxima
    return 1 + int(interior.sum())


def critical_jump_mean(delta, Lam, b, a_lo=-30.0, a_hi=-1.0):
    """Bisection for a*, the jump mean where a crash mode first appears."""
    if n_modes(delta, Lam, a_lo, b) == 1:
        return np.nan                                           # never bimodal in range
    if n_modes(delta, Lam, a_hi, b) == 2:
        return a_hi                                             # always bimodal in range
    for _ in range(40):
        mid = 0.5 * (a_lo + a_hi)
        if n_modes(delta, Lam, mid, b) == 2:
            a_lo = mid
        else:
            a_hi = mid
    return 0.5 * (a_lo + a_hi)


# 4. Simulation

def simulate(p, n_days, n_paths, seed=RNG_SEED):
    """Daily log returns (n_days x n_paths) and variance paths."""
    rng = np.random.default_rng(seed)
    s, m, eps, d, rho, Lam, a, b = (p[k] for k in ("s", "m", "eps", "delta", "rho", "Lam", "a", "b"))
    r = np.empty((n_days, n_paths)); Vs = np.empty((n_days, n_paths))   # storage for results
    if d > 0:
        V = stats.gamma(1 / d, scale=d).rvs(n_paths, random_state=rng)   # start in the stationary law
        c = d * (1 - np.exp(-eps)) / 2                                   # CIR scale factor
        df = 2 / d                                                       # chi-squared degrees of freedom
        sig_v = np.sqrt(2 * eps * d)                                     # vol of variance
    else:
        V = np.ones(n_paths)                                             # constant variance
    for t in range(n_days):                                              # one step per trading day
        if d > 0:
            V_next = c * rng.noncentral_chisquare(df, V * np.exp(-eps) / c)   # exact CIR step, always > 0
            Vbar = 0.5 * (V + V_next)                                         # average variance over the day
            corr_part = (rho / sig_v) * (V_next - V - eps * (1 - Vbar))      # shock shared with V (leverage)
            diff_part = np.sqrt(1 - rho ** 2) * np.sqrt(Vbar) * rng.standard_normal(n_paths)
        else:
            V_next = V; Vbar = V
            corr_part = 0.0
            diff_part = rng.standard_normal(n_paths)
        dX = (m - s * Vbar / 2) + corr_part + diff_part                      # drift + diffusion
        if Lam > 0:
            jumps = rng.random(n_paths) < Lam                                # jump today?
            dX = dX + jumps * rng.normal(a, b, n_paths)                      # add jump size
        r[t] = s * dX                                                        # back to log returns
        Vs[t] = V_next
        V = V_next                                                           # move to the next day
    return r, Vs


# Stylised facts

def acf_at(x, lags):
    """Autocorrelation of x at the given lags."""
    x = x - x.mean(); v = np.dot(x, x)            # centre, then total variance
    return np.array([np.dot(x[:-k], x[k:]) / v for k in lags])


def facts(r):
    """Stylised-fact statistics for one return series."""
    z = (r - r.mean()) / r.std(ddof=1)            # sigma moves
    pc = np.expm1(r)                              # percentage changes
    ev = np.where(pc <= -0.03)[0]; w = np.diff(ev)   # gaps between 3% falls
    return dict(
        ann_vol=r.std(ddof=1) * np.sqrt(252) * 100,
        skew=stats.skew(r),
        exkurt=stats.kurtosis(r),
        acf_r1=acf_at(r, [1])[0],
        acf_abs1=acf_at(np.abs(r), [1])[0],
        acf_abs100=acf_at(np.abs(r), [100])[0],
        leverage1=np.corrcoef(r[:-1], np.abs(r[1:]))[0, 1],
        n_z3=int((z < -3).sum()),
        n_z5=int((z < -5).sum()),
        n_fall5pct=int((pc <= -0.05).sum()),
        recur5=np.mean(w <= 5) if len(w) else np.nan,     # next 3% fall within 5 days
        worst_pct=pc.min() * 100,
    )


# S&P 500, 4 Jan 1950 to 16 Oct 2026 (19,319 returns); overwritten if a file is given
OBSERVED = dict(ann_vol=15.80, skew=-0.948, exkurt=25.20, acf_r1=-0.002, acf_abs1=0.269,
                acf_abs100=0.094, leverage1=-0.108, n_z3=155, n_z5=34, n_fall5pct=30,
                recur5=0.420, worst_pct=-20.47)


def load_returns(path):
    """Daily log returns from the yfinance spreadsheet."""
    raw = pd.read_excel(path, header=None, skiprows=3, usecols=[0, 1]).dropna()
    raw.columns = ["date", "close"]                 # date and close columns only
    close = raw.set_index(pd.to_datetime(raw["date"]))["close"].astype(float).sort_index()
    return np.log(close).diff().dropna().values    # r_t = ln S_t - ln S_(t-1)


# Figures

def fig_phase_portrait(p):
    """Skeleton trajectories in (X~, u)."""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    s_vis = 0.2                                   # enlarged s so the drift is visible
    tau = np.linspace(0, 6 / p["eps"], 400)       # six e-folding times
    for u0 in (-0.8, -0.4, 0.5, 1.0, 2.0, 3.0):
        u = u0 * np.exp(-p["eps"] * tau)                                   # exact solution
        Xt = -(s_vis / (2 * p["eps"])) * u0 * (1 - np.exp(-p["eps"] * tau))
        ax.plot(Xt, u, lw=1.6)
        ax.annotate("", xy=(Xt[40], u[40]), xytext=(Xt[30], u[30]),
                    arrowprops=dict(arrowstyle="->", lw=1.2))              # direction arrow
    ax.axhline(0, c="k", lw=2, label="line of equilibria, u = 0 (V = 1)")
    ax.set_xlabel("detrended log price X~ (s exaggerated to 0.2)")
    ax.set_ylabel("variance deviation u = V − 1")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "s1_phase_portrait.png", dpi=150); plt.close(fig)


def fig_variance_bifurcation():
    """Stationary variance densities and the mode against δ."""
    v = np.linspace(1e-3, 4, 800)                  # variance grid
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for d in (0.25, 0.57, 1.0, 2.0, 4.0):
        ax[0].plot(v, variance_density(v, d), label=f"δ = {d}")
    ax[0].set_ylim(0, 2.2); ax[0].set_xlabel("V"); ax[0].set_ylabel("stationary density p(V)"); ax[0].legend()
    ds = np.linspace(0.01, 3, 300)                 # δ values for the mode plot
    ax[1].plot(ds, [variance_mode(d) for d in ds], lw=2)
    ax[1].axvline(1, ls="--", c="grey"); ax[1].text(1.05, 0.8, "δ = 1\n(Feller: 2κθ = ξ²)", fontsize=9)
    ax[1].set_xlabel("δ"); ax[1].set_ylabel("most likely variance (mode of V)")
    fig.tight_layout(); fig.savefig(FIG / "s2_variance_bifurcation.png", dpi=150); plt.close(fig)


def fig_crash_mode(curves, p):
    """Critical jump mean a*(Λ) and example one-day densities."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for d, (Ls, astar) in curves.items():
        ax[0].semilogx(Ls, astar, lw=2, label=f"δ = {d}")
    ax[0].plot(p["Lam"], p["a"], "k*", ms=12, label="screening values")
    ax[0].set_xlabel("jump probability per day Λ"); ax[0].set_ylabel("critical jump mean a*")
    ax[0].set_title("bimodal (crash mode) below each curve", fontsize=10); ax[0].legend(fontsize=8)
    x = np.linspace(-14, 4, 1200)                  # one-day moves in daily sigmas
    for d in (0.0, 1.0):
        for a in (-5.42, -9.0):
            f = one_day_density(x, d, p["Lam"], a, p["b"])
            ax[1].semilogy(x, f, label=f"δ = {d}, a = {a}")
    ax[1].set_ylim(1e-7, 1); ax[1].set_xlabel("one-day move ΔX (daily σ)"); ax[1].set_ylabel("density")
    ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "s3_crash_mode_bifurcation.png", dpi=150); plt.close(fig)


def fig_paths(sims):
    """One simulated return path per model."""
    fig, ax = plt.subplots(len(sims), 1, figsize=(11, 2.0 * len(sims)), sharex=True, sharey=True)
    for i, (name, (r, _)) in enumerate(sims.items()):
        ax[i].plot(np.arange(len(r)) / 252, r[:, 0] * 100, lw=0.4)   # first path, in per cent
        ax[i].set_ylabel(f"{name}\nr, %")
    ax[-1].set_xlabel("years")
    fig.tight_layout(); fig.savefig(FIG / "s4_simulated_paths.png", dpi=150); plt.close(fig)


def fig_compare(table):
    """Simulated medians with 5-95% bands against the S&P 500."""
    keys = [("exkurt", "excess kurtosis"), ("acf_abs1", "ACF of |r|, lag 1"), ("leverage1", "corr(r_t, |r_t+1|)"),
            ("n_z3", "days with z < −3"), ("n_z5", "days with z < −5"), ("recur5", "3% falls recurring within 5 days")]
    fig, ax = plt.subplots(2, 3, figsize=(12, 6.5))
    models = [m for m in table.index.get_level_values(0).unique()]
    for k, (key, label) in enumerate(keys):
        a = ax.flat[k]
        med = [table.loc[(mname, "median"), key] for mname in models]
        lo = [table.loc[(mname, "p05"), key] for mname in models]
        hi = [table.loc[(mname, "p95"), key] for mname in models]
        xs = np.arange(len(models))                # one x position per model
        a.errorbar(xs, med, yerr=[np.subtract(med, lo), np.subtract(hi, med)], fmt="o", capsize=4)
        a.axhline(OBSERVED[key], c="C3", ls="--", label="S&P 500")
        a.set_xticks(xs); a.set_xticklabels(models); a.set_title(label, fontsize=10)
    ax.flat[0].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "s5_model_vs_data.png", dpi=150); plt.close(fig)


# Main

def main():
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    if len(sys.argv) > 1:
        OBSERVED.update(facts(load_returns(sys.argv[1])))       # use fresh data statistics
    p = params("Bates")

    # 1. Skeleton
    sk = skeleton_analysis(p)
    print("1. Deterministic skeleton")
    print("   Jacobian\n", sk["J"])
    print("   eigenvalues", sk["eigenvalues"], " characteristic roots", sk["char_roots"])
    print(f"   variance half-life {sk['half_life_days']:.1f} days; "
          f"permanent X~ shift per unit u0 {sk['permanent_shift_per_unit_u0']:.3f} daily sigmas")

    # 2. Moments and linearisation
    ms = moment_system(p)
    print("\n2. Moment system, δ =", p["delta"])
    print("   eigenvalues", ms["eigenvalues"])
    print("   stationary moments", ms["stationary"], " Gamma check", ms["gamma_check"])
    print("   one-day excess kurtosis without jumps = 3δ =", 3 * p["delta"])
    print("\n   Linearised (OU) against exact (Gamma) stationary law")
    print(linearisation_check().to_string(float_format=lambda v: f"{v:.4g}"))

    # 3. Bifurcations
    print("\n3a. Mode of V:", {d: variance_mode(d) for d in (0.25, 0.57, 1.0, 2.0)})
    print("\n3b. Crash-mode bifurcation")
    for d in (0.0, 0.57, 1.0):
        print(f"   δ = {d}: modes at screening values = {n_modes(d, p['Lam'], p['a'], p['b'])}, "
              f"a* = {critical_jump_mean(d, p['Lam'], p['b']):.2f}")
    Ls = np.logspace(-4, -1, 25)                                                 # jump rates to scan
    curves = {d: (Ls, np.array([critical_jump_mean(d, L, p['b']) for L in Ls])) for d in (0.0, 0.57, 1.0)}
    for d, (L, astar) in curves.items():
        print(f"   δ = {d}: a*(Λ) at Λ = 1e-4, 1e-3, 1e-2, 1e-1:",
              np.round(np.interp(np.log10([1e-4, 1e-3, 1e-2, 1e-1]), np.log10(L), astar), 2))

    # 4. Simulation
    n_days, n_paths = 19_319, 200                     # same length as the S&P 500 sample
    print(f"\n4. Simulation: {n_paths} paths of {n_days} days per model")
    sims, rows = {}, []
    for name in SPECS:
        r, V = simulate(params(name), n_days, n_paths)
        sims[name] = (r, V)
        F = pd.DataFrame([facts(r[:, j]) for j in range(n_paths)])          # facts per path
        for q, lab in ((0.05, "p05"), (0.5, "median"), (0.95, "p95")):
            rows.append(pd.Series(F.quantile(q), name=(name, lab)))
        print(f"   {name} done")
    table = pd.DataFrame(rows)                        # rows: model x quantile
    table.index = pd.MultiIndex.from_tuples(table.index)
    obs = pd.DataFrame([OBSERVED], index=pd.MultiIndex.from_tuples([("S&P 500", "observed")]))
    print(pd.concat([table, obs]).to_string(float_format=lambda v: f"{v:.3g}"))

    # Figures
    fig_phase_portrait(p); fig_variance_bifurcation(); fig_crash_mode(curves, p)
    fig_paths(sims); fig_compare(table)
    print(f"\nFigures saved to {FIG.resolve()}")


if __name__ == "__main__":
    main()
