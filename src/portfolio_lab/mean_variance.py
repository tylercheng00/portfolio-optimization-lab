"""Mean-variance (Markowitz) portfolio optimization with CVXPY."""
from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

Bounds = tuple[float, float]


@dataclass
class PortfolioResult:
    """Weights of an optimized portfolio plus its model statistics."""

    weights: np.ndarray
    expected_return: float
    volatility: float
    status: str
    info: dict = field(default_factory=dict)


def portfolio_stats(weights: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> tuple[float, float]:
    """Model expected return and volatility of a weight vector."""
    w = np.asarray(weights, dtype=float)
    return float(w @ mu), float(np.sqrt(max(w @ cov @ w, 0.0)))


def _check(mu: np.ndarray | None, cov: np.ndarray, bounds: Bounds) -> int:
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    if cov.shape != (n, n):
        raise ValueError("cov must be a square matrix")
    if mu is not None and np.shape(mu) != (n,):
        raise ValueError("mu must have one entry per asset")
    lo, hi = bounds
    if lo > hi or n * lo > 1 + 1e-12 or n * hi < 1 - 1e-12:
        raise ValueError(f"bounds {bounds} make a fully invested portfolio infeasible")
    return n


def _budget_constraints(w: cp.Variable, bounds: Bounds) -> list:
    lo, hi = bounds
    return [cp.sum(w) == 1, w >= lo, w <= hi]


def _solve(problem: cp.Problem) -> str:
    problem.solve(solver=cp.CLARABEL)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"optimization failed with status {problem.status!r}")
    return problem.status


def _result(w: np.ndarray, mu: np.ndarray | None, cov: np.ndarray, status: str, **info) -> PortfolioResult:
    w = np.where(np.abs(w) < 1e-10, 0.0, w)
    ret, vol = portfolio_stats(w, mu if mu is not None else np.zeros(len(w)), cov)
    return PortfolioResult(weights=w, expected_return=ret, volatility=vol, status=status, info=info)


def min_variance(cov: np.ndarray, mu: np.ndarray | None = None, bounds: Bounds = (0.0, 1.0)) -> PortfolioResult:
    """Global minimum-variance portfolio."""
    n = _check(mu, cov, bounds)
    w = cp.Variable(n)
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), _budget_constraints(w, bounds))
    status = _solve(prob)
    return _result(w.value, mu, cov, status)


def target_return(
    mu: np.ndarray, cov: np.ndarray, target: float, bounds: Bounds = (0.0, 1.0)
) -> PortfolioResult:
    """Minimum-variance portfolio whose model expected return is at least ``target``."""
    n = _check(mu, cov, bounds)
    w = cp.Variable(n)
    cons = _budget_constraints(w, bounds) + [mu @ w >= target]
    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(cov))), cons)
    status = _solve(prob)
    return _result(w.value, mu, cov, status, target=target)


def risk_aversion(
    mu: np.ndarray, cov: np.ndarray, gamma: float, bounds: Bounds = (0.0, 1.0)
) -> PortfolioResult:
    """Maximize ``mu'w - gamma * w'Σw`` (risk-aversion form of mean-variance)."""
    if gamma < 0:
        raise ValueError("gamma must be non-negative")
    n = _check(mu, cov, bounds)
    w = cp.Variable(n)
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(cov)))
    prob = cp.Problem(objective, _budget_constraints(w, bounds))
    status = _solve(prob)
    return _result(w.value, mu, cov, status, gamma=gamma)


def max_sharpe(
    mu: np.ndarray, cov: np.ndarray, risk_free: float = 0.0, bounds: Bounds = (0.0, 1.0)
) -> PortfolioResult:
    """Maximum Sharpe-ratio portfolio via the standard convex reformulation.

    Substituting ``y = κ w`` turns the ratio into a convex QP:
    minimize ``y'Σy`` subject to ``(mu - r_f)'y = 1``, ``sum(y) = κ``, ``κ·lo <= y <= κ·hi``.
    """
    n = _check(mu, cov, bounds)
    excess = np.asarray(mu, dtype=float) - risk_free
    if np.max(excess) <= 0:
        raise ValueError("at least one asset needs an expected return above the risk-free rate")
    lo, hi = bounds
    y = cp.Variable(n)
    kappa = cp.Variable(nonneg=True)
    cons = [excess @ y == 1, cp.sum(y) == kappa, y >= lo * kappa, y <= hi * kappa]
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(cov))), cons)
    status = _solve(prob)
    res = _result(y.value / kappa.value, mu, cov, status, risk_free=risk_free)
    res.info["sharpe"] = (res.expected_return - risk_free) / res.volatility
    return res


@dataclass
class Frontier:
    """Points on the efficient frontier, ordered by increasing risk."""

    returns: np.ndarray
    volatilities: np.ndarray
    weights: np.ndarray  # shape (n_points, n_assets)


def efficient_frontier(
    mu: np.ndarray, cov: np.ndarray, n_points: int = 30, bounds: Bounds = (0.0, 1.0)
) -> Frontier:
    """Trace the efficient frontier between the minimum-variance and maximum-return portfolios."""
    n = _check(mu, cov, bounds)
    w = cp.Variable(n)
    best = cp.Problem(cp.Maximize(mu @ w), _budget_constraints(w, bounds))
    _solve(best)
    r_max = float(mu @ w.value)
    r_min = min_variance(cov, mu, bounds).expected_return

    targets = np.linspace(r_min, r_max - 1e-6 * max(abs(r_max), 1.0), n_points)
    pts = [target_return(mu, cov, t, bounds) for t in targets]
    return Frontier(
        returns=np.array([p.expected_return for p in pts]),
        volatilities=np.array([p.volatility for p in pts]),
        weights=np.vstack([p.weights for p in pts]),
    )
