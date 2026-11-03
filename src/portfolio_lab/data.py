"""Synthetic market data.

Every number produced here is generated from a random factor model.
No real prices, tickers, accounts or client data are used anywhere in this project.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TRADING_DAYS = 252


@dataclass(frozen=True)
class SyntheticMarket:
    """A simulated set of daily asset returns and the parameters that generated them."""

    returns: np.ndarray  # shape (n_periods, n_assets), daily simple returns
    names: tuple[str, ...]
    true_mu: np.ndarray  # annualized expected returns used by the generator
    true_cov: np.ndarray  # annualized covariance used by the generator

    @property
    def n_assets(self) -> int:
        return self.returns.shape[1]


def synthetic_market(
    n_assets: int = 8,
    n_periods: int = 1000,
    n_factors: int = 2,
    seed: int | None = 0,
) -> SyntheticMarket:
    """Simulate daily returns from a linear factor model.

    Covariance is ``B F B' + D`` (factor loadings ``B``, factor covariance ``F``,
    diagonal idiosyncratic variance ``D``), which guarantees a well-conditioned,
    positive-definite matrix. Expected returns loosely increase with volatility.
    """
    if n_assets < 2:
        raise ValueError("n_assets must be at least 2")
    rng = np.random.default_rng(seed)

    loadings = rng.normal(0.0, 1.0, size=(n_assets, n_factors))
    loadings[:, 0] = np.abs(loadings[:, 0]) + 0.5  # a common "market" factor
    factor_vol = np.linspace(0.16, 0.08, n_factors)
    factor_cov = np.diag(factor_vol**2)
    idio_vol = rng.uniform(0.06, 0.20, size=n_assets)

    cov = loadings @ factor_cov @ loadings.T + np.diag(idio_vol**2)
    vol = np.sqrt(np.diag(cov))
    mu = 0.01 + 0.35 * vol + rng.normal(0.0, 0.015, size=n_assets)

    returns = rng.multivariate_normal(mu / TRADING_DAYS, cov / TRADING_DAYS, size=n_periods)
    names = tuple(f"Asset {chr(ord('A') + i)}" if i < 26 else f"Asset {i + 1}" for i in range(n_assets))
    return SyntheticMarket(returns=returns, names=names, true_mu=mu, true_cov=cov)
