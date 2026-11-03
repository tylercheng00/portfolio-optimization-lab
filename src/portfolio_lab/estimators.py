"""Estimators for expected returns and covariance."""
from __future__ import annotations

import numpy as np

from .data import TRADING_DAYS


def _as_2d(returns: np.ndarray) -> np.ndarray:
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2 or x.shape[0] < 2:
        raise ValueError("returns must be a 2-D array with at least two rows (periods x assets)")
    return x


def mean_returns(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> np.ndarray:
    """Annualized arithmetic mean of periodic returns."""
    return _as_2d(returns).mean(axis=0) * periods_per_year


def sample_covariance(returns: np.ndarray, periods_per_year: int = TRADING_DAYS) -> np.ndarray:
    """Annualized sample covariance (unbiased)."""
    return np.cov(_as_2d(returns), rowvar=False, ddof=1) * periods_per_year


def ledoit_wolf_covariance(
    returns: np.ndarray, periods_per_year: int = TRADING_DAYS
) -> tuple[np.ndarray, float]:
    """Ledoit-Wolf (2004) shrinkage towards a scaled identity.

    Returns the annualized shrunk covariance and the shrinkage intensity in [0, 1].
    """
    x = _as_2d(returns)
    t, n = x.shape
    x = x - x.mean(axis=0)
    s = x.T @ x / t
    m = np.trace(s) / n
    target = m * np.eye(n)

    d2 = np.sum((s - target) ** 2)
    # (1/T^2) * sum_t ||x_t x_t' - S||_F^2, computed without forming T outer products
    b_bar2 = (np.sum(np.sum(x**2, axis=1) ** 2) - t * np.sum(s**2)) / t**2
    b2 = min(b_bar2, d2)
    shrinkage = 0.0 if d2 == 0 else float(b2 / d2)

    cov = shrinkage * target + (1.0 - shrinkage) * s
    return cov * periods_per_year, shrinkage
