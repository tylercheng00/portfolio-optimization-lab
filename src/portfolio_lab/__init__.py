"""Portfolio Optimization Lab — mean-variance and risk-parity optimization with CVXPY.

For research and educational purposes only. Not investment advice.
"""
from .admm import ADMMResult, admm_portfolio_qp, mean_variance_admm
from .data import SyntheticMarket, synthetic_market
from .estimators import ledoit_wolf_covariance, mean_returns, sample_covariance
from .mean_variance import (
    Frontier,
    PortfolioResult,
    efficient_frontier,
    max_sharpe,
    min_variance,
    portfolio_stats,
    risk_aversion,
    target_return,
)
from .rebalance import RebalanceResult, rebalance
from .risk_parity import CCDResult, risk_contributions, risk_parity, risk_parity_ccd

__version__ = "0.1.0"

__all__ = [
    "ADMMResult",
    "CCDResult",
    "Frontier",
    "PortfolioResult",
    "RebalanceResult",
    "SyntheticMarket",
    "admm_portfolio_qp",
    "efficient_frontier",
    "ledoit_wolf_covariance",
    "max_sharpe",
    "mean_returns",
    "mean_variance_admm",
    "min_variance",
    "portfolio_stats",
    "rebalance",
    "risk_aversion",
    "risk_contributions",
    "risk_parity",
    "risk_parity_ccd",
    "sample_covariance",
    "synthetic_market",
    "target_return",
]
