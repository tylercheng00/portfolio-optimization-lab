"""A fixed-iteration ADMM kernel for the portfolio QP — software reference for hardware research.

Research direction: low-latency rebalancing needs a solver whose run time is *predictable*.
An OSQP-style ADMM iteration with a pre-factorized KKT matrix uses only dense
matrix-vector products, additions and clamps, with a fixed iteration count. That
structure maps naturally onto pipelined hardware (FPGA / ASIC).

This module is a plain NumPy reference model. It also lets you emulate reduced numeric
precision (``float32`` or a fixed-point rounding step) to study how word length affects
convergence. **No hardware implementation exists in this repository.**

Problem solved (risk-aversion mean-variance with box constraints)::

    minimize   ½ x'Px + q'x
    subject to sum(x) = 1,   lower <= x <= upper

with ``P = 2γΣ`` and ``q = −mu``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class ADMMResult:
    x: np.ndarray
    iterations: int
    primal_residual: np.ndarray  # ‖Ax − z‖∞ per iteration
    dual_residual: np.ndarray  # ‖ρA'(z − z_prev)‖∞ per iteration
    x_history: np.ndarray  # iterate x after each iteration


def _quantizer(precision: str, frac_bits: int | None):
    dtype = {"float64": np.float64, "float32": np.float32}[precision]
    if frac_bits is None:
        return dtype, (lambda v: v.astype(dtype, copy=False))
    scale = float(2**frac_bits)
    return dtype, (lambda v: (np.round(v * scale) / scale).astype(dtype, copy=False))


def admm_portfolio_qp(
    P: np.ndarray,
    q: np.ndarray,
    lower: float | np.ndarray = 0.0,
    upper: float | np.ndarray = 1.0,
    rho: float = 1.0,
    sigma: float = 1e-6,
    alpha: float = 1.6,
    n_iter: int = 200,
    precision: str = "float64",
    frac_bits: int | None = None,
) -> ADMMResult:
    """Run exactly ``n_iter`` ADMM iterations (no early exit, for deterministic latency).

    ``precision`` selects ``float64`` or ``float32`` arithmetic; ``frac_bits`` additionally
    rounds every state vector to a multiple of ``2**-frac_bits`` to emulate fixed-point storage.
    """
    P = np.asarray(P, dtype=float)
    q = np.asarray(q, dtype=float)
    n = len(q)
    dtype, quant = _quantizer(precision, frac_bits)

    # Constraint matrix A = [1'; I] with bounds l <= Ax <= u
    A = np.vstack([np.ones((1, n)), np.eye(n)])
    lo = np.concatenate([[1.0], np.broadcast_to(lower, n)])
    hi = np.concatenate([[1.0], np.broadcast_to(upper, n)])

    # One-time "factorization" (here an explicit inverse): done offline, not per iteration.
    K = P + sigma * np.eye(n) + rho * A.T @ A
    M = np.linalg.inv(K).astype(dtype)
    A_d, At_d = A.astype(dtype), A.T.astype(dtype)
    q_d, lo_d, hi_d = q.astype(dtype), lo.astype(dtype), hi.astype(dtype)
    rho_d, sig_d, a_d = dtype(rho), dtype(sigma), dtype(alpha)

    x = np.full(n, 1.0 / n, dtype=dtype)
    z = A_d @ x
    y = np.zeros(n + 1, dtype=dtype)
    r_prim, r_dual, xs = [], [], []

    for _ in range(n_iter):
        x_t = quant(M @ (sig_d * x - q_d + At_d @ (rho_d * z - y)))
        z_t = A_d @ x_t
        x = quant(a_d * x_t + (1 - a_d) * x)
        z_relax = a_d * z_t + (1 - a_d) * z
        z_new = quant(np.clip(z_relax + y / rho_d, lo_d, hi_d))
        y = quant(y + rho_d * (z_relax - z_new))
        r_dual.append(float(np.max(np.abs(rho_d * (At_d @ (z_new - z))))))
        z = z_new
        r_prim.append(float(np.max(np.abs(A_d @ x - z))))
        xs.append(x.astype(float))

    return ADMMResult(
        x=x.astype(float),
        iterations=n_iter,
        primal_residual=np.array(r_prim),
        dual_residual=np.array(r_dual),
        x_history=np.array(xs),
    )


def mean_variance_admm(mu: np.ndarray, cov: np.ndarray, gamma: float, **kwargs) -> ADMMResult:
    """Convenience wrapper: maximize ``mu'w − γ w'Σw`` (long-only by default) with ADMM."""
    return admm_portfolio_qp(2.0 * gamma * np.asarray(cov), -np.asarray(mu), **kwargs)
