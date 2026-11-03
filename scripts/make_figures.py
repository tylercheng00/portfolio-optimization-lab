"""Generate the animated figures in docs/assets and the data for the interactive demo.

Run:  pip install -e ".[figures]"  &&  python scripts/make_figures.py

Every figure is produced by running this library on a *synthetic* market
(see portfolio_lab.data). Nothing here is real market data or a track record.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

import portfolio_lab as pl  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "docs" / "assets"
DATA = ROOT / "docs" / "data"

BG, PANEL, INK, MUTED, GRID = "#0b1d33", "#10233d", "#e9edf3", "#93a2b5", "#1f3550"
GOLD, GOLD2 = "#c9a35b", "#e2c88f"
PALETTE = ["#c9a35b", "#4f86c6", "#5fb39a", "#d9775a", "#9b86d6", "#e2c88f", "#6fa8dc", "#b5c46a"]
FPS = 12
# Latin text in DejaVu Sans; Chinese falls back to any installed CJK font.
plt.rcParams["font.family"] = ["DejaVu Sans", "Noto Sans CJK TC", "Noto Sans TC", "PingFang TC",
                               "Microsoft JhengHei", "WenQuanYi Zen Hei"]


def style(ax, title=None):
    ax.set_facecolor(PANEL)
    for s in ax.spines.values():
        s.set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    if title:
        ax.set_title(title, color=INK, fontsize=10.5, loc="left", pad=8, fontweight="semibold")


def figure(w=9.6, h=4.6):
    fig = plt.figure(figsize=(w, h), dpi=80, facecolor=BG)
    fig.text(0.99, 0.015, "Synthetic data · model output, not investment advice",
             ha="right", va="bottom", color=MUTED, fontsize=7.5, alpha=0.9)
    return fig


def pct(ax, axis="both"):
    fmt = matplotlib.ticker.PercentFormatter(1.0, decimals=0)
    if axis in ("x", "both"):
        ax.xaxis.set_major_formatter(fmt)
    if axis in ("y", "both"):
        ax.yaxis.set_major_formatter(fmt)


def ease(t):
    return t * t * (3 - 2 * t)


# --------------------------------------------------------------------------- data
market = pl.synthetic_market(n_assets=8, n_periods=1000, seed=0)
names = [n.replace("Asset ", "") for n in market.names]
mu = pl.mean_returns(market.returns)
cov, _ = pl.ledoit_wolf_covariance(market.returns)
vols = np.sqrt(np.diag(cov))
n = len(mu)

front = pl.efficient_frontier(mu, cov, n_points=40)
ms = pl.max_sharpe(mu, cov)
mv = pl.min_variance(cov, mu)
rng = np.random.default_rng(42)
rand_w = rng.dirichlet(np.full(n, 0.6), size=1500)
rand_r = rand_w @ mu
rand_v = np.sqrt(np.einsum("ij,jk,ik->i", rand_w, cov, rand_w))


# --------------------------------------------------------------------------- 1. frontier
def make_frontier_gif():
    fig = figure()
    ax = fig.add_axes([0.07, 0.13, 0.52, 0.76])
    bx = fig.add_axes([0.66, 0.13, 0.31, 0.76])
    style(ax, "Efficient frontier  ·  效率前緣")
    style(bx, "Weights along the frontier  ·  權重")
    ax.set_xlabel("Model volatility", color=MUTED, fontsize=8.5)
    ax.set_ylabel("Model expected return", color=MUTED, fontsize=8.5)
    pct(ax)
    ax.set_xlim(0, vols.max() * 1.12)
    ax.set_ylim(min(mu.min(), rand_r.min()) - 0.01, mu.max() * 1.12)

    cloud = ax.scatter(rand_v, rand_r, s=4, c=MUTED, alpha=0.0, linewidths=0)
    for i in range(n):
        ax.scatter(vols[i], mu[i], s=36, color=PALETTE[i], zorder=3, edgecolor=BG, linewidth=0.8)
        ax.annotate(names[i], (vols[i], mu[i]), xytext=(5, 3), textcoords="offset points", color=INK, fontsize=8)
    (line,) = ax.plot([], [], color=GOLD2, lw=2.4, zorder=4)
    (dot,) = ax.plot([], [], "o", color=GOLD, ms=9, mec=INK, mew=1.2, zorder=5)
    star = ax.scatter([], [], marker="*", s=180, color=GOLD2, edgecolor=BG, zorder=6)
    lbl = ax.text(0.02, 0.96, "", transform=ax.transAxes, color=INK, fontsize=8.5, va="top")

    bars = bx.bar(names, front.weights[0], color=PALETTE[:n])
    bx.set_ylim(0, 1)
    pct(bx, "y")

    n_pts = len(front.returns)
    k_cloud, k_trace, k_hold = 10, 50, 14
    frames = k_cloud + k_trace + k_hold

    k_star = 12
    frames += k_star

    def update(f):
        if f < k_cloud:
            cloud.set_alpha(0.35 * ease((f + 1) / k_cloud))
            lbl.set_text("1,500 random long-only portfolios")
            return
        if f >= k_cloud + k_trace + k_hold:
            a = ease(min((f - k_cloud - k_trace - k_hold + 1) / k_star, 1.0))
            v = (1 - a) * front.volatilities[-1] + a * ms.volatility
            r = (1 - a) * front.returns[-1] + a * ms.expected_return
            dot.set_data([v], [r])
            for b, h0, h1 in zip(bars, front.weights[-1], ms.weights):
                b.set_height((1 - a) * h0 + a * h1)
            lbl.set_text("★ max model Sharpe ratio: its weights")
            return
        t = min((f - k_cloud) / (k_trace - 1), 1.0)
        pos = ease(t) * (n_pts - 1)
        j = int(np.floor(pos))
        a = pos - j
        j2 = min(j + 1, n_pts - 1)
        v = (1 - a) * front.volatilities[j] + a * front.volatilities[j2]
        r = (1 - a) * front.returns[j] + a * front.returns[j2]
        w = (1 - a) * front.weights[j] + a * front.weights[j2]
        line.set_data(np.r_[front.volatilities[: j + 1], v], np.r_[front.returns[: j + 1], r])
        dot.set_data([v], [r])
        for b, h in zip(bars, w):
            b.set_height(h)
        lbl.set_text(f"CVXPY: min variance s.t. return ≥ {r:.1%}")
        if f >= k_cloud + k_trace:
            star.set_offsets([[ms.volatility, ms.expected_return]])
            lbl.set_text("★ maximum model Sharpe ratio portfolio")

    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(ASSETS / "efficient_frontier.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)


# --------------------------------------------------------------------------- 2. risk parity
def make_risk_parity_gif():
    ccd = pl.risk_parity_ccd(cov, init=np.full(n, 1 / n), tol=1e-12)
    hist, whist = ccd.history, ccd.weight_history
    k = min(len(hist) - 1, 8)
    steps_per = 7
    fig = figure()
    ax = fig.add_axes([0.07, 0.13, 0.42, 0.76])
    bx = fig.add_axes([0.56, 0.13, 0.41, 0.76])
    style(ax, "Risk contribution per asset  ·  風險貢獻")
    style(bx, "Weights  ·  權重")
    pct(ax, "y")
    pct(bx, "y")
    ymax = max(hist.max(), 1 / n) * 1.25
    ax.set_ylim(0, ymax)
    bx.set_ylim(0, max(whist.max(), 1 / n) * 1.25)
    ax.axhline(1 / n, color=GOLD2, lw=1.4, ls="--")
    ax.text(n - 0.5, 1 / n, " target 1/N", color=GOLD2, fontsize=8, va="bottom", ha="right")
    rb = ax.bar(names, hist[0], color=PALETTE[:n])
    wb = bx.bar(names, whist[0], color=PALETTE[:n], alpha=0.9)
    lbl = ax.text(0.02, 0.96, "", transform=ax.transAxes, color=INK, fontsize=8.5, va="top")

    frames = 8 + k * steps_per + 14

    def update(f):
        if f < 8:
            s, a = 0, 0.0
        else:
            g = min(f - 8, k * steps_per)
            s, a = divmod(g, steps_per)
            a = ease(a / steps_per)
            if s >= k:
                s, a = k - 1, 1.0
        h = (1 - a) * hist[s] + a * hist[min(s + 1, k)]
        w = (1 - a) * whist[s] + a * whist[min(s + 1, k)]
        for b, v in zip(rb, h):
            b.set_height(v)
        for b, v in zip(wb, w):
            b.set_height(v)
        it = 0 if f < 8 else min(s + (1 if a > 0.5 else 0), k)
        spread = np.abs(h - 1 / n).max()
        lbl.set_text(("Start: equal weights" if it == 0 else f"Coordinate-descent sweep {it}")
                     + f"   ·   max deviation {spread:.2%}")

    anim = FuncAnimation(fig, update, frames=frames)
    anim.save(ASSETS / "risk_parity.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return ccd


# --------------------------------------------------------------------------- 3. ADMM
def make_admm_gif():
    gamma, iters = 3.0, 200
    ref = pl.risk_aversion(mu, cov, gamma).weights
    runs = {
        "float64": pl.mean_variance_admm(mu, cov, gamma, n_iter=iters),
        "float32": pl.mean_variance_admm(mu, cov, gamma, n_iter=iters, precision="float32"),
        "fixed-point (12 frac bits)": pl.mean_variance_admm(mu, cov, gamma, n_iter=iters, frac_bits=12),
    }
    base = runs["float64"]
    fig = figure()
    ax = fig.add_axes([0.07, 0.13, 0.42, 0.76])
    bx = fig.add_axes([0.57, 0.13, 0.40, 0.76])
    style(ax, "ADMM iterates → CVXPY optimum  ·  迭代收斂")
    style(bx, "Primal residual by precision  ·  精度比較")
    ax.set_xlim(0, iters)
    ax.set_ylim(0, max(base.x_history.max(), ref.max()) * 1.15)
    pct(ax, "y")
    ax.set_xlabel("Iteration (fixed count)", color=MUTED, fontsize=8.5)
    bx.set_xlim(0, iters)
    bx.set_yscale("log")
    allr = np.concatenate([r.primal_residual for r in runs.values()])
    bx.set_ylim(max(allr[allr > 0].min() * 0.5, 1e-12), allr.max() * 2)
    bx.set_xlabel("Iteration", color=MUTED, fontsize=8.5)
    for i in range(n):
        ax.axhline(ref[i], color=PALETTE[i], lw=0.8, ls=":", alpha=0.7)
    wl = [ax.plot([], [], color=PALETTE[i], lw=1.8)[0] for i in range(n)]
    cols = [GOLD2, "#4f86c6", "#d9775a"]
    rl = [bx.plot([], [], color=c, lw=1.8, label=k)[0] for c, k in zip(cols, runs)]
    leg = bx.legend(loc="upper right", fontsize=7.5, facecolor=PANEL, edgecolor=GRID)
    for t in leg.get_texts():
        t.set_color(INK)
    lbl = ax.text(0.02, 0.96, "", transform=ax.transAxes, color=INK, fontsize=8.5, va="top")
    ax.text(0.98, 0.04, "dotted lines = CVXPY solution", transform=ax.transAxes, color=MUTED, fontsize=7.5,
            ha="right", va="bottom", bbox=dict(facecolor=PANEL, edgecolor="none", pad=2))

    frames, hold = 56, 12

    def update(f):
        m = max(1, int(round(ease(min(f / (frames - 1), 1.0)) * iters)))
        xs = np.arange(1, m + 1)
        for i in range(n):
            wl[i].set_data(xs, base.x_history[:m, i])
        for line, r in zip(rl, runs.values()):
            line.set_data(xs, np.maximum(r.primal_residual[:m], 1e-16))
        gap = np.abs(base.x_history[m - 1] - ref).max()
        lbl.set_text(f"iteration {m:3d}   ·   max |w − w*| = {gap:.1e}")

    anim = FuncAnimation(fig, update, frames=frames + hold)
    anim.save(ASSETS / "admm_convergence.gif", writer=PillowWriter(fps=FPS))
    plt.close(fig)
    return runs, ref


# --------------------------------------------------------------------------- demo data
def r(a, d=5):
    return np.round(np.asarray(a, dtype=float), d).tolist()


if __name__ == "__main__":
    ASSETS.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    make_frontier_gif()
    ccd = make_risk_parity_gif()
    runs, ref = make_admm_gif()

    gammas = np.geomspace(0.5, 40, 36)
    ra = [pl.risk_aversion(mu, cov, g) for g in gammas]
    rp = pl.risk_parity(cov)
    demo = {
        "note": "Synthetic data generated by portfolio_lab.synthetic_market(seed=0). Model output only.",
        "names": names,
        "mu": r(mu),
        "cov": r(cov, 6),
        "vols": r(vols),
        "frontier": {"vol": r(front.volatilities), "ret": r(front.returns), "w": r(front.weights, 4)},
        "max_sharpe": {"vol": round(ms.volatility, 5), "ret": round(ms.expected_return, 5), "w": r(ms.weights, 4)},
        "min_variance": {"vol": round(mv.volatility, 5), "ret": round(mv.expected_return, 5), "w": r(mv.weights, 4)},
        "risk_aversion": {"gamma": r(gammas, 4), "w": [r(x.weights, 4) for x in ra],
                          "vol": [round(x.volatility, 5) for x in ra], "ret": [round(x.expected_return, 5) for x in ra]},
        "risk_parity": {"w": r(rp, 4), "ccd_rc": r(ccd.history, 5), "ccd_w": r(ccd.weight_history, 5)},
        "admm": {"ref": r(ref, 5), "x": r(runs["float64"].x_history[::2], 5),
                 "res": {k: r(np.maximum(v.primal_residual, 1e-16), 12) for k, v in runs.items()}},
    }
    (DATA / "demo.json").write_text(json.dumps(demo, separators=(",", ":")))
    for p in sorted(ASSETS.glob("*.gif")):
        print(f"{p.name:28s} {p.stat().st_size / 1024:7.0f} KB")
    print(f"demo.json {(DATA / 'demo.json').stat().st_size / 1024:.0f} KB")
