"""Quickstart: optimize portfolios on a synthetic market.

Run:  python examples/quickstart.py
All data is randomly generated; results describe the model, not any real market.
"""
import numpy as np

import portfolio_lab as pl

market = pl.synthetic_market(n_assets=8, n_periods=1000, seed=0)
mu = pl.mean_returns(market.returns)
cov, shrink = pl.ledoit_wolf_covariance(market.returns)
print(f"Ledoit-Wolf shrinkage intensity: {shrink:.3f}\n")


def show(title, w):
    ret, vol = pl.portfolio_stats(w, mu, cov)
    print(f"{title:<22} model return {ret:6.2%}   model vol {vol:6.2%}")
    print("   " + "  ".join(f"{n[-1]}:{x:5.1%}" for n, x in zip(market.names, w)))


show("Minimum variance", pl.min_variance(cov, mu).weights)
show("Max Sharpe (model)", pl.max_sharpe(mu, cov).weights)
show("Risk aversion γ=3", pl.risk_aversion(mu, cov, gamma=3.0).weights)
rp = pl.risk_parity(cov)
show("Risk parity", rp)
print("   risk contributions:", np.round(pl.risk_contributions(rp, cov), 3))

current = np.full(market.n_assets, 1 / market.n_assets)
reb = pl.rebalance(current, mu, cov, gamma=3.0, cost=0.002)
print(f"\nRebalance from equal weight with 0.2% cost: turnover {reb.turnover:.1%}")

admm = pl.mean_variance_admm(mu, cov, gamma=3.0, n_iter=300)
gap = np.abs(admm.x - pl.risk_aversion(mu, cov, 3.0).weights).max()
print(f"ADMM reference kernel (300 fixed iterations): max weight gap vs CVXPY {gap:.1e}")
