"""Risk-parity / risk-budgeting portfolios.

A risk-budgeting portfolio sets each asset's share of total portfolio risk equal to a
target budget ``b_i`` (equal budgets give "risk parity"). The long-only problem has a
convex formulation (Spinu, 2013):

    minimize   ½ y'Σy − Σ_i b_i log(y_i)      over y > 0,     then  w = y / sum(y)
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np


def risk_contributions(weights: np.ndarray, cov: np.ndarray, relative: bool = True) -> np.ndarray:
    """Each asset's contribution ``w_i (Σw)_i`` to portfolio variance.

    With ``relative=True`` the contributions are fractions of total variance and sum to 1.
    """
    w = np.asarray(weights, dtype=float)
    rc = w * (cov @ w)
    return rc / rc.sum() if relative else rc


def _budgets(n: int, budgets: np.ndarray | None) -> np.ndarray:
    b = np.full(n, 1.0 / n) if budgets is None else np.asarray(budgets, dtype=float)
    if b.shape != (n,) or np.any(b <= 0):
        raise ValueError("budgets must be positive, one per asset")
    return b / b.sum()


def risk_parity(cov: np.ndarray, budgets: np.ndarray | None = None) -> np.ndarray:
    """Long-only risk-budgeting weights solved with CVXPY (exponential-cone program)."""
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    b = _budgets(n, budgets)
    y = cp.Variable(n)
    objective = cp.Minimize(0.5 * cp.quad_form(y, cp.psd_wrap(cov)) - b @ cp.log(y))
    prob = cp.Problem(objective)
    prob.solve(solver=cp.CLARABEL)
    if prob.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"risk parity failed with status {prob.status!r}")
    y_val = np.maximum(y.value, 0.0)
    return y_val / y_val.sum()


@dataclass
class CCDResult:
    weights: np.ndarray
    iterations: int
    history: np.ndarray  # relative risk contributions after each sweep, shape (iters+1, n)
    weight_history: np.ndarray  # weights after each sweep, shape (iters+1, n)


def risk_parity_ccd(
    cov: np.ndarray,
    budgets: np.ndarray | None = None,
    tol: float = 1e-10,
    max_iter: int = 500,
    init: np.ndarray | None = None,
) -> CCDResult:
    """Cyclical coordinate descent on the same convex objective (Griveau-Billion et al., 2013).

    Each coordinate update is a closed-form root of ``Σ_ii y_i² + c_i y_i − b_i = 0``,
    so the method needs only multiply-adds and one square root per asset — a useful
    baseline for studying fixed-latency implementations.
    """
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    b = _budgets(n, budgets)
    diag = np.diag(cov)
    # start from `init` weights if given, otherwise inverse volatility
    y = 1.0 / np.sqrt(diag) if init is None else np.asarray(init, dtype=float).copy()
    if y.shape != (n,) or np.any(y <= 0):
        raise ValueError("init must be positive, one entry per asset")
    y /= np.sqrt(y @ cov @ y)

    w0 = y / y.sum()
    hist = [risk_contributions(w0, cov)]
    whist = [w0]
    it = 0
    for it in range(1, max_iter + 1):
        y_prev = y.copy()
        for i in range(n):
            c = cov[i] @ y - diag[i] * y[i]
            y[i] = (-c + np.sqrt(c * c + 4.0 * diag[i] * b[i])) / (2.0 * diag[i])
        w = y / y.sum()
        hist.append(risk_contributions(w, cov))
        whist.append(w)
        if np.max(np.abs(y - y_prev)) < tol * max(1.0, np.max(np.abs(y))):
            break
    return CCDResult(weights=y / y.sum(), iterations=it, history=np.array(hist), weight_history=np.array(whist))
