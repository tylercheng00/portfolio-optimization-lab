import numpy as np
import pytest

import portfolio_lab as pl


@pytest.fixture(scope="module")
def market():
    m = pl.synthetic_market(n_assets=6, n_periods=1500, seed=1)
    mu = pl.mean_returns(m.returns)
    cov, _ = pl.ledoit_wolf_covariance(m.returns)
    return mu, cov


def _is_budget(w, lo=0.0, hi=1.0, tol=1e-6):
    return abs(w.sum() - 1) < tol and w.min() >= lo - tol and w.max() <= hi + tol


# ---------------- data & estimators ----------------
def test_synthetic_market_shapes_and_determinism():
    a = pl.synthetic_market(n_assets=5, n_periods=300, seed=7)
    b = pl.synthetic_market(n_assets=5, n_periods=300, seed=7)
    assert a.returns.shape == (300, 5)
    assert np.allclose(a.returns, b.returns)
    assert np.all(np.linalg.eigvalsh(a.true_cov) > 0)


def test_ledoit_wolf_is_psd_and_shrinkage_in_range(market):
    m = pl.synthetic_market(n_assets=10, n_periods=60, seed=3)
    cov, s = pl.ledoit_wolf_covariance(m.returns)
    assert 0.0 <= s <= 1.0
    assert np.all(np.linalg.eigvalsh(cov) > 0)
    assert np.allclose(cov, cov.T)


# ---------------- mean-variance ----------------
def test_min_variance_is_lowest_risk(market):
    mu, cov = market
    mv = pl.min_variance(cov, mu)
    assert _is_budget(mv.weights)
    for gamma in (0.5, 2.0, 10.0):
        assert pl.risk_aversion(mu, cov, gamma).volatility >= mv.volatility - 1e-6


def test_target_return_meets_target(market):
    mu, cov = market
    target = float(np.median(mu))
    res = pl.target_return(mu, cov, target)
    assert _is_budget(res.weights)
    assert res.expected_return >= target - 1e-6


def test_weight_bounds_respected(market):
    mu, cov = market
    res = pl.risk_aversion(mu, cov, gamma=0.2, bounds=(0.05, 0.4))
    assert _is_budget(res.weights, 0.05, 0.4)


def test_infeasible_bounds_rejected(market):
    mu, cov = market
    with pytest.raises(ValueError):
        pl.min_variance(cov, mu, bounds=(0.0, 0.1))  # 6 assets x 10% < 100%


def test_max_sharpe_beats_frontier_points(market):
    mu, cov = market
    ms = pl.max_sharpe(mu, cov)
    assert _is_budget(ms.weights)
    front = pl.efficient_frontier(mu, cov, n_points=15)
    sharpe = front.returns / front.volatilities
    assert ms.info["sharpe"] >= sharpe.max() - 1e-4


def test_frontier_is_monotone(market):
    mu, cov = market
    f = pl.efficient_frontier(mu, cov, n_points=12)
    assert np.all(np.diff(f.returns) > -1e-8)
    assert np.all(np.diff(f.volatilities) > -1e-6)
    assert all(_is_budget(w) for w in f.weights)


# ---------------- risk parity ----------------
def test_risk_parity_equalizes_contributions(market):
    _, cov = market
    w = pl.risk_parity(cov)
    rc = pl.risk_contributions(w, cov)
    assert _is_budget(w)
    assert np.allclose(rc, 1 / len(w), atol=1e-4)


def test_risk_budgets(market):
    _, cov = market
    b = np.array([3, 1, 1, 1, 2, 2], dtype=float)
    w = pl.risk_parity(cov, budgets=b)
    assert np.allclose(pl.risk_contributions(w, cov), b / b.sum(), atol=1e-4)


def test_ccd_matches_cvxpy(market):
    _, cov = market
    ccd = pl.risk_parity_ccd(cov)
    assert np.allclose(ccd.weights, pl.risk_parity(cov), atol=1e-5)
    assert ccd.history.shape[0] == ccd.iterations + 1


# ---------------- rebalance ----------------
def test_rebalance_zero_cost_matches_risk_aversion(market):
    mu, cov = market
    n = len(mu)
    r = pl.rebalance(np.full(n, 1 / n), mu, cov, gamma=3.0, cost=0.0)
    assert np.allclose(r.weights, pl.risk_aversion(mu, cov, 3.0).weights, atol=1e-4)


def test_rebalance_high_cost_means_no_trade(market):
    mu, cov = market
    n = len(mu)
    current = np.full(n, 1 / n)
    r = pl.rebalance(current, mu, cov, gamma=3.0, cost=10.0)
    assert r.turnover < 1e-6


def test_rebalance_turnover_cap(market):
    mu, cov = market
    n = len(mu)
    r = pl.rebalance(np.full(n, 1 / n), mu, cov, gamma=1.0, max_turnover=0.2)
    assert r.turnover <= 0.2 + 1e-6


# ---------------- ADMM reference kernel ----------------
def test_admm_matches_cvxpy(market):
    mu, cov = market
    gamma = 3.0
    ref = pl.risk_aversion(mu, cov, gamma).weights
    res = pl.mean_variance_admm(mu, cov, gamma, n_iter=600)
    assert np.allclose(res.x, ref, atol=1e-3)
    assert res.primal_residual[-1] < 1e-4
    assert len(res.primal_residual) == 600


def test_admm_reduced_precision_still_close(market):
    mu, cov = market
    ref = pl.risk_aversion(mu, cov, 3.0).weights
    f32 = pl.mean_variance_admm(mu, cov, 3.0, n_iter=600, precision="float32")
    fx = pl.mean_variance_admm(mu, cov, 3.0, n_iter=600, frac_bits=16)
    assert np.allclose(f32.x, ref, atol=2e-3)
    assert np.allclose(fx.x, ref, atol=5e-3)


def test_ccd_from_equal_weights_converges(market):
    _, cov = market
    n = cov.shape[0]
    ccd = pl.risk_parity_ccd(cov, init=np.full(n, 1 / n))
    assert np.allclose(ccd.history[-1], 1 / n, atol=1e-6)
    assert np.abs(ccd.history[0] - 1 / n).max() > np.abs(ccd.history[-1] - 1 / n).max()
