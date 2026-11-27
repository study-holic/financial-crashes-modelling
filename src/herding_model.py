"""
Herding model (RQ2): equilibria, stability, bifurcation, and finite-N simulation.

Model: N traders, each optimistic (+) or pessimistic (-).
Opinion index x = (n+ - n-) / N, between -1 and 1.
Mean-field ODE, time τ = 2νt:
    dx/dτ = cosh(αx + β) [tanh(αx + β) - x]
α = herding strength, β = bias from news or fundamentals.

Usage:
    pip install numpy pandas scipy matplotlib
    python src/herding_model.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import optimize, integrate

FIG = Path("figures"); FIG.mkdir(exist_ok=True)   # figures go here
RNG_SEED = 20261106                               # fixed seed so runs repeat


# 1. Mean-field ODE

def F(x, alpha, beta):
    """Right-hand side dx/dτ."""
    U = alpha * x + beta                          # switching incentive
    return np.cosh(U) * (np.tanh(U) - x)


def F_prime(x, alpha, beta):
    """dF/dx, used for linear stability."""
    U = alpha * x + beta
    return alpha * np.sinh(U) * (np.tanh(U) - x) + np.cosh(U) * (alpha / np.cosh(U) ** 2 - 1)


# 2. Equilibria and stability

def equilibria(alpha, beta, n_grid=4001):
    """All roots of x = tanh(αx + β) in [-1, 1]."""
    g = lambda x: np.tanh(alpha * x + beta) - x   # zero exactly at equilibria
    xs = np.linspace(-1, 1, n_grid)
    gs = g(xs)
    roots = []
    for i in range(n_grid - 1):
        if gs[i] == 0:
            roots.append(xs[i])                   # grid point is a root
        elif gs[i] * gs[i + 1] < 0:
            roots.append(optimize.brentq(g, xs[i], xs[i + 1]))   # sign change: refine
    return np.array(sorted(roots))


def eigenvalue(x_star, alpha, beta):
    """Linearised growth rate at an equilibrium: cosh(U*)(α(1 - x*²) - 1)."""
    U = alpha * x_star + beta
    return np.cosh(U) * (alpha * (1 - x_star ** 2) - 1)


def classify(alpha, beta):
    """Table of equilibria with eigenvalue and stability."""
    rows = []
    for x in equilibria(alpha, beta):
        lam = eigenvalue(x, alpha, beta)
        rows.append(dict(alpha=alpha, beta=beta, x_star=x, eigenvalue=lam,
                         check_Fprime=F_prime(x, alpha, beta),      # should equal eigenvalue
                         stable=lam < 0))
    return pd.DataFrame(rows)


# 3. Bifurcations

def pitchfork_amplitude(alpha):
    """Leading-order herding level near α = 1: x* ≈ sqrt(3(α - 1) / α³)."""
    return np.sqrt(3 * np.maximum(alpha - 1, 0) / alpha ** 3)   # from tanh(αx) ≈ αx - (αx)³/3


def fold_points(alpha):
    """Saddle-node points (x_f, β_f) for α > 1: x = tanh(αx + β) and α(1 - x²) = 1."""
    xf = np.sqrt(1 - 1 / alpha)                   # from the second condition
    bf = np.arctanh(xf) - alpha * xf              # from the first condition
    return xf, bf                                 # fold at (x_f, β_f); mirror at (-x_f, -β_f)


def beta_critical(alpha):
    """Half-width of the bistable band |β| < β_c (zero for α ≤ 1)."""
    if alpha <= 1:
        return 0.0
    return -fold_points(alpha)[1]


def crash_size(alpha):
    """Jump in x when the optimistic branch disappears at β = -β_c."""
    xf, bf = fold_points(alpha)                   # optimistic branch ends at (β_f, x_f)
    lower = equilibria(alpha, bf - 1e-9).min()    # the pessimistic state it falls to
    return xf, lower, xf - lower


def finite_N_threshold(N, beta=0.0):
    """Smallest α at which the exact stationary law of n+ is bimodal."""
    def bimodal(alpha):
        n, pi = stationary_distribution(N, alpha, beta)
        return ((pi[1:-1] > pi[:-2]) & (pi[1:-1] > pi[2:])).sum() >= 2   # two peaks
    lo, hi = 0.5, 1.5
    for _ in range(40):                           # bisection on α
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if bimodal(mid) else (mid, hi)
    return hi


def hysteresis_run(alpha, beta_max=0.6, rate=2e-4, x0=None):
    """Integrate the ODE while β falls slowly then rises again."""
    T = 2 * beta_max / rate                       # time for one sweep
    beta_of = lambda t: beta_max - rate * t if t < T else -beta_max + rate * (t - T)
    rhs = lambda t, x: F(x, alpha, beta_of(t))
    x0 = equilibria(alpha, beta_max).max() if x0 is None else x0   # start on the optimistic branch
    sol = integrate.solve_ivp(rhs, (0, 2 * T), [x0], max_step=5, rtol=1e-8, atol=1e-10)
    betas = np.array([beta_of(t) for t in sol.t])
    return sol.t, sol.y[0], betas


# 4. Finite N: birth-death chain

def rates(n, N, alpha, beta):
    """Switching rates per unit τ: up = pessimist -> optimist, down = optimist -> pessimist."""
    x = (2 * n - N) / N                           # opinion index from the count n = n+
    U = alpha * x + beta
    up = (N - n) * np.exp(U) / 2                  # each pessimist switches at e^U / 2
    down = n * np.exp(-U) / 2                     # each optimist switches at e^-U / 2
    return up, down


def stationary_distribution(N, alpha, beta):
    """Exact stationary law of n+ from detailed balance."""
    n = np.arange(N + 1)
    up, down = rates(n, N, alpha, beta)
    logpi = np.zeros(N + 1)
    logpi[1:] = np.cumsum(np.log(up[:-1]) - np.log(down[1:]))   # π(n+1)/π(n) = up(n)/down(n+1)
    pi = np.exp(logpi - logpi.max())
    return n, pi / pi.sum()


def mean_crash_time(N, alpha, beta=0.0):
    """Exact mean time to fall from the optimistic well to the pessimistic one."""
    n, pi = stationary_distribution(N, alpha, beta)
    _, down = rates(n, N, alpha, beta)
    xs = equilibria(alpha, beta)
    i = int(round(N * (1 + xs.max()) / 2))        # start: optimistic equilibrium
    j = int(round(N * (1 + xs.min()) / 2))        # target: pessimistic equilibrium
    tail = np.cumsum(pi[::-1])[::-1]              # tail[k] = P(n >= k)
    return sum(tail[k] / (pi[k] * down[k]) for k in range(j + 1, i + 1))   # sum of one-step times


def barrier(alpha):
    """Large-N barrier ΔΦ, with π ∝ exp(-N φ(x)) and φ'(x) = artanh(x) - αx (β = 0)."""
    xs = equilibria(alpha, 0.0).max()
    phi = lambda x: x * np.arctanh(x) + 0.5 * np.log(1 - x ** 2) - alpha * x ** 2 / 2
    return phi(0.0) - phi(xs)                     # height of the hill at x = 0


def gillespie(N, alpha, beta, t_max, n0=None, seed=RNG_SEED):
    """Exact stochastic simulation of n+ (Gillespie algorithm)."""
    rng = np.random.default_rng(seed)
    n = N // 2 if n0 is None else n0
    t, ts, ns = 0.0, [0.0], [n]
    while t < t_max:
        up, down = rates(n, N, alpha, beta)
        total = up + down
        t += rng.exponential(1 / total)           # waiting time to the next switch
        n += 1 if rng.random() < up / total else -1   # which way the switch goes
        ts.append(t); ns.append(n)
    ts, ns = np.array(ts), np.array(ns)
    return ts, (2 * ns - N) / N                   # times and opinion index


def simulated_crash_times(N, alpha, n_runs, seed=RNG_SEED):
    """Monte Carlo check of mean_crash_time."""
    rng = np.random.default_rng(seed)
    xs = equilibria(alpha, 0.0)
    i = int(round(N * (1 + xs.max()) / 2)); j = int(round(N * (1 + xs.min()) / 2))
    out = []
    for _ in range(n_runs):
        n, t = i, 0.0
        while n > j:                              # run until the pessimistic well is reached
            up, down = rates(n, N, alpha, 0.0)
            total = up + down
            t += rng.exponential(1 / total)
            n += 1 if rng.random() < up / total else -1
        out.append(t)
    return np.array(out)


# Figures

def fig_pitchfork():
    """x* against α at β = 0."""
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    al = np.linspace(0.01, 3, 400)
    up = np.array([equilibria(a, 0).max() for a in al])          # upper branch (= 0 for α ≤ 1)
    ax.plot(al[al <= 1], np.zeros((al <= 1).sum()), "C0", lw=2, label="stable")
    ax.plot(al[al > 1], np.zeros((al > 1).sum()), "C0--", lw=1.5, label="unstable")
    ax.plot(al[al > 1], up[al > 1], "C0", lw=2); ax.plot(al[al > 1], -up[al > 1], "C0", lw=2)
    ax.plot(al[al > 1], pitchfork_amplitude(al[al > 1]), "C1:", label="leading order √(3(α−1)/α³)")
    ax.set_ylim(-1.05, 1.05); ax.set_xlabel("herding strength α"); ax.set_ylabel("equilibrium opinion x*")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "h1_pitchfork.png", dpi=150); plt.close(fig)


def fig_imperfect(alpha=1.5):
    """x* against β at fixed α: S-curve, folds, and a hysteresis run."""
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    for b in np.linspace(-0.6, 0.6, 600):
        for x in equilibria(alpha, b):
            ax.plot(b, x, ".", ms=1.5, c="C0" if eigenvalue(x, alpha, b) < 0 else "C3")   # blue stable, red unstable
    t, x, betas = hysteresis_run(alpha)
    ax.plot(betas, x, "k", lw=0.9, alpha=0.7, label="slow β sweep (ODE)")
    xf, bf = fold_points(alpha)
    ax.plot([bf, -bf], [xf, -xf], "ks", ms=6, label="folds")
    ax.set_xlabel("bias β"); ax.set_ylabel("opinion x"); ax.set_title(f"α = {alpha}", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "h2_imperfect_hysteresis.png", dpi=150); plt.close(fig)


def fig_cusp():
    """Bistable region in the (α, β) plane."""
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    al = np.linspace(1.0001, 3, 300)
    bc = np.array([beta_critical(a) for a in al])
    ax.fill_between(al, -bc, bc, alpha=0.3, label="bistable: two stable states")
    ax.plot(al, bc, "C0"); ax.plot(al, -bc, "C0")
    ax.plot(1, 0, "ko", label="cusp (α, β) = (1, 0)")
    ax.set_xlim(0, 3); ax.set_ylim(-1.2, 1.2)
    ax.set_xlabel("herding strength α"); ax.set_ylabel("bias β")
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "h3_cusp.png", dpi=150); plt.close(fig)


def fig_finite_N(N=100):
    """Gillespie paths and exact stationary laws for finite N."""
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for alpha, c in ((0.8, "C0"), (1.3, "C1")):
        t, x = gillespie(N, alpha, 0.0, t_max=3000)
        ax[0].plot(t, x, c, lw=0.5, label=f"α = {alpha}")
    ax[0].set_xlabel("τ"); ax[0].set_ylabel("opinion x"); ax[0].legend(fontsize=8)
    for alpha in (0.8, 1.0, 1.2, 1.5):
        n, pi = stationary_distribution(N, alpha, 0.0)
        ax[1].plot((2 * n - N) / N, pi * N / 2, label=f"α = {alpha}")   # rescale to a density in x
    ax[1].set_xlabel("opinion x"); ax[1].set_ylabel(f"stationary density (N = {N})"); ax[1].legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIG / "h4_finite_N.png", dpi=150); plt.close(fig)


def fig_crash_times(Ns, T, alpha, slope, dphi):
    """Mean crash time against N on a log scale."""
    fig, ax = plt.subplots(figsize=(6.5, 4.3))
    ax.semilogy(Ns, T, "o-", label="exact mean time")
    ax.semilogy(Ns, T[0] * np.exp(dphi * (np.array(Ns) - Ns[0])), "--", label=f"slope NΔΦ, ΔΦ = {dphi:.4f}")
    ax.set_xlabel("number of traders N"); ax.set_ylabel("mean time between crashes (τ units)")
    ax.set_title(f"α = {alpha}, β = 0; fitted slope {slope:.4f}", fontsize=10)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(FIG / "h5_crash_times.png", dpi=150); plt.close(fig)


# Main

def main():
    pd.set_option("display.width", 200)
    fmt = lambda v: f"{v:.4g}"

    # Equilibria and stability at sample points
    print("1. Equilibria and stability")
    tab = pd.concat([classify(a, b) for a, b in ((0.5, 0), (1.0, 0), (1.5, 0), (2.0, 0), (1.5, 0.1), (1.5, 0.3))])
    print(tab.to_string(index=False, float_format=fmt))

    # Pitchfork at α = 1 (β = 0)
    print("\n2. Pitchfork: exact x* against √(3(α−1)/α³)")
    for a in (1.01, 1.05, 1.1, 1.3, 1.5, 2.0):
        print(f"   α = {a}: x* = {equilibria(a, 0).max():.4f}, approx {pitchfork_amplitude(a):.4f}")

    # Folds, bistable band, and crash size
    print("\n3. Folds and hysteresis (β ≠ 0)")
    for a in (1.2, 1.5, 2.0, 3.0):
        xf, lower, dx = crash_size(a)
        print(f"   α = {a}: β_c = {beta_critical(a):.4f}, optimistic branch ends at x = {xf:.4f}, "
              f"falls to x = {lower:.4f}, crash size Δx = {dx:.4f}")
    t, x, betas = hysteresis_run(1.5)
    k = np.argmax(x < 0)                           # first crossing of x = 0 on the way down
    print(f"   ODE sweep α = 1.5 (rate 2e-4): x crosses 0 at β = {betas[k]:.4f}; fold at {-beta_critical(1.5):.4f}")

    # Finite N
    print("\n4. Finite N, α = 1.5, β = 0")
    dphi = barrier(1.5)
    Ns = [20, 40, 60, 80, 100, 150, 200]
    T = np.array([mean_crash_time(N, 1.5) for N in Ns])
    slope = np.polyfit(Ns[2:], np.log(T[2:]), 1)[0]   # growth rate of ln T in N
    for N, Ti in zip(Ns, T):
        print(f"   N = {N}: mean crash time {Ti:.4g}")
    print(f"   fitted d(ln T)/dN = {slope:.4f}; barrier ΔΦ = {dphi:.4f}")
    sim = simulated_crash_times(30, 1.5, 400)
    print(f"   check N = 30: exact {mean_crash_time(30, 1.5):.1f}, simulated {sim.mean():.1f} ± {sim.std() / np.sqrt(len(sim)):.1f}")
    for N in (25, 50, 100, 200, 400, 1000):
        print(f"   N = {N}: stationary law bimodal for α > {finite_N_threshold(N):.4f}")
    for a in (0.8, 1.0, 1.2, 1.5):
        n, pi = stationary_distribution(100, a, 0.0)
        modes = ((2 * n[1:-1] - 100) / 100)[(pi[1:-1] > pi[:-2]) & (pi[1:-1] > pi[2:])]
        print(f"   N = 100, α = {a}: stationary modes at x = {np.round(modes, 2)}")

    # Figures
    fig_pitchfork(); fig_imperfect(); fig_cusp(); fig_finite_N(); fig_crash_times(Ns, T, 1.5, slope, dphi)
    print(f"\nFigures saved to {FIG.resolve()}")


if __name__ == "__main__":
    main()
