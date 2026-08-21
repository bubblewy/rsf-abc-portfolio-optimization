"""Deterministic finance baselines."""

from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .objective import OmegaObjective
from .portfolio import bounded_simplex_projection, repair_cardinality
from .types import OptimizationResult


def equal_weight_n(objective: OmegaObjective) -> OptimizationResult:
    start = time.perf_counter()
    weights = np.full(objective.returns.shape[1], 1.0 / objective.returns.shape[1])
    stats = objective.evaluate(weights)
    return OptimizationResult(
        method="ew_n",
        weights=weights,
        objective=stats.omega,
        evaluations=1,
        runtime_seconds=time.perf_counter() - start,
        convergence_evaluations=[1],
        convergence_best=[stats.omega],
        diagnostics={"constraint_matched": False, "turnover": stats.turnover},
    )


def equal_weight_k(
    objective: OmegaObjective,
    cardinality: int,
    lower: float,
    upper: float,
) -> OptimizationResult:
    start = time.perf_counter()
    n_assets = objective.returns.shape[1]
    individual_scores = np.empty(n_assets)
    means = objective.returns.mean(axis=0)
    for asset in range(n_assets):
        singleton = np.zeros(n_assets)
        singleton[asset] = 1.0
        individual_scores[asset] = objective.evaluate(singleton).omega
    order = np.lexsort((np.arange(n_assets), -means, -individual_scores))
    support = order[:cardinality]
    weights = np.zeros(n_assets)
    weights[support] = bounded_simplex_projection(
        np.full(cardinality, 1.0 / cardinality), lower, upper
    )
    stats = objective.evaluate(weights)
    return OptimizationResult(
        method="ew_k",
        weights=weights,
        objective=stats.omega,
        evaluations=n_assets + 1,
        runtime_seconds=time.perf_counter() - start,
        convergence_evaluations=[n_assets + 1],
        convergence_best=[stats.omega],
        diagnostics={"selected_indices": support.tolist(), "turnover": stats.turnover},
    )


def _minimum_variance_weights(
    covariance: np.ndarray,
    lower: float,
    upper: float,
    initial: np.ndarray,
) -> tuple[np.ndarray, bool, str]:
    n = covariance.shape[0]
    result = minimize(
        lambda w: float(w @ covariance @ w),
        x0=initial,
        jac=lambda w: 2.0 * covariance @ w,
        bounds=[(lower, upper)] * n,
        constraints={"type": "eq", "fun": lambda w: float(w.sum() - 1.0)},
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-12, "disp": False},
    )
    return np.asarray(result.x, dtype=float), bool(result.success), str(result.message)


def minimum_variance_k(
    objective: OmegaObjective,
    cardinality: int,
    lower: float,
    upper: float,
) -> OptimizationResult:
    start = time.perf_counter()
    returns = objective.returns
    n_assets = returns.shape[1]
    covariance = np.cov(returns, rowvar=False, ddof=1)
    ridge = max(float(np.trace(covariance)) / max(n_assets, 1), 1e-12) * 1e-8
    covariance = covariance + np.eye(n_assets) * ridge

    continuous, success_all, message_all = _minimum_variance_weights(
        covariance,
        0.0,
        1.0,
        np.full(n_assets, 1.0 / n_assets),
    )
    order = np.lexsort((np.arange(n_assets), -continuous))
    support = order[:cardinality]
    support_covariance = covariance[np.ix_(support, support)]
    initial = bounded_simplex_projection(
        np.maximum(continuous[support], 0.0), lower, upper
    )
    selected, success_support, message_support = _minimum_variance_weights(
        support_covariance,
        lower,
        upper,
        initial,
    )
    if not success_support:
        selected = bounded_simplex_projection(initial, lower, upper)
    weights = np.zeros(n_assets)
    weights[support] = selected
    weights = repair_cardinality(weights, cardinality, lower, upper)
    stats = objective.evaluate(weights)
    success = success_all and success_support
    return OptimizationResult(
        method="mv_k",
        weights=weights,
        objective=stats.omega,
        evaluations=0,
        runtime_seconds=time.perf_counter() - start,
        convergence_evaluations=[0],
        convergence_best=[stats.omega],
        success=success,
        message=f"continuous={message_all}; support={message_support}",
        diagnostics={
            "selected_indices": support.tolist(),
            "variance": float(weights @ covariance @ weights),
            "turnover": stats.turnover,
        },
    )
