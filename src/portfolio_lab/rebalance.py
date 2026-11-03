"""Rebalancing with a proportional transaction-cost penalty."""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np

from .mean_variance import Bounds


@dataclass
class RebalanceResult:
    weights: np.ndarray
    trades: np.ndarray  # new weights minus current weights
    turnover: float  # sum of absolute trades
    status: str


def rebalance(
    current: np.ndarray,
    mu: np.ndarray,
    cov: np.ndarray,
    gamma: float,
    cost: float = 0.0,
    bounds: Bounds = (0.0, 1.0),
    max_turnover: float | None = None,
) -> RebalanceResult:
    """Maximize ``mu'w − γ w'Σw − cost·‖w − current‖₁``.

    ``cost`` is a proportional cost per unit of weight traded; an optional
    ``max_turnover`` caps ``‖w − current‖₁``.
    """
    w0 = np.asarray(current, dtype=float)
    n = len(w0)
    lo, hi = bounds
    w = cp.Variable(n)
    trade = w - w0
    objective = cp.Maximize(mu @ w - gamma * cp.quad_form(w, cp.psd_wrap(cov)) - cost * cp.norm1(trade))
    cons = [cp.sum(w) == 1, w >= lo, w <= hi]
    if max_turnover is not None:
        cons.append(cp.norm1(trade) <= max_turnover)
    prob = cp.Problem(objective, cons)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"rebalance failed with status {prob.status!r}")
    new = np.where(np.abs(w.value) < 1e-10, 0.0, w.value)
    t = new - w0
    t = np.where(np.abs(t) < 1e-8, 0.0, t)
    return RebalanceResult(weights=new, trades=t, turnover=float(np.abs(t).sum()), status=prob.status)
