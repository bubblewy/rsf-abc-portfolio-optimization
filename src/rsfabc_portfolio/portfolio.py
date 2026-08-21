"""Portfolio feasibility, turnover, and drift operations."""

from __future__ import annotations

import numpy as np


def bounded_simplex_projection(
    values: np.ndarray,
    lower: float,
    upper: float,
    target: float = 1.0,
    tolerance: float = 1e-12,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0:
        raise ValueError("values must be a non-empty one-dimensional array")
    n = values.size
    if lower < 0 or upper <= lower:
        raise ValueError("bounds must satisfy 0 <= lower < upper")
    if n * lower > target + tolerance or n * upper < target - tolerance:
        raise ValueError("bounded simplex is infeasible")

    lo = float(np.min(values - upper))
    hi = float(np.max(values - lower))
    for _ in range(200):
        midpoint = 0.5 * (lo + hi)
        projected = np.clip(values - midpoint, lower, upper)
        total = float(projected.sum())
        if abs(total - target) <= tolerance:
            break
        if total > target:
            lo = midpoint
        else:
            hi = midpoint
    projected = np.clip(values - 0.5 * (lo + hi), lower, upper)

    residual = target - float(projected.sum())
    if abs(residual) > tolerance:
        free = np.flatnonzero(
            (projected > lower + tolerance) & (projected < upper - tolerance)
        )
        if free.size:
            projected[free] += residual / free.size
        else:
            direction = projected < upper - tolerance if residual > 0 else projected > lower + tolerance
            candidates = np.flatnonzero(direction)
            if not candidates.size:
                raise RuntimeError("projection residual cannot be allocated")
            projected[candidates[0]] += residual
    return projected


def repair_cardinality(
    scores: np.ndarray,
    cardinality: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be one-dimensional")
    n_assets = scores.size
    if not 1 <= cardinality <= n_assets:
        raise ValueError("cardinality must be between 1 and number of assets")
    if cardinality * lower > 1.0 + 1e-12 or cardinality * upper < 1.0 - 1e-12:
        raise ValueError("cardinality and bounds are infeasible")
    safe_scores = np.nan_to_num(scores, nan=-np.inf, posinf=1e12, neginf=-1e12)
    order = np.lexsort((np.arange(n_assets), -safe_scores))
    support = order[:cardinality]
    selected = safe_scores[support].copy()
    finite = np.isfinite(selected)
    if not finite.all():
        selected[~finite] = np.min(selected[finite]) if finite.any() else 0.0
    weights = np.zeros(n_assets, dtype=float)
    weights[support] = bounded_simplex_projection(selected, lower, upper)
    return weights


def random_feasible_population(
    rng: np.random.Generator,
    population_size: int,
    n_assets: int,
    cardinality: int,
    lower: float,
    upper: float,
) -> np.ndarray:
    population = np.zeros((population_size, n_assets), dtype=float)
    for row in range(population_size):
        support = np.sort(rng.choice(n_assets, size=cardinality, replace=False))
        raw = rng.exponential(scale=1.0, size=cardinality)
        population[row, support] = bounded_simplex_projection(raw, lower, upper)
    return population


def is_feasible(
    weights: np.ndarray,
    cardinality: int,
    lower: float,
    upper: float,
    tolerance: float = 1e-8,
) -> bool:
    weights = np.asarray(weights, dtype=float)
    active = weights > tolerance
    if weights.ndim != 1 or int(active.sum()) != cardinality:
        return False
    if abs(float(weights.sum()) - 1.0) > tolerance:
        return False
    if np.any(weights < -tolerance):
        return False
    selected = weights[active]
    return bool(
        np.all(selected >= lower - tolerance) and np.all(selected <= upper + tolerance)
    )


def turnover_one_way(
    target: np.ndarray,
    pretrade: np.ndarray,
    cash_pre: float = 0.0,
) -> float:
    target = np.asarray(target, dtype=float)
    pretrade = np.asarray(pretrade, dtype=float)
    if target.shape != pretrade.shape:
        raise ValueError("target and pretrade weights must share a shape")
    if cash_pre < -1e-12:
        raise ValueError("cash_pre cannot be negative")
    return 0.5 * (float(np.abs(target - pretrade).sum()) + abs(float(cash_pre)))


def drift_weights(target: np.ndarray, realized_asset_returns: np.ndarray) -> np.ndarray:
    target = np.asarray(target, dtype=float)
    realized_asset_returns = np.asarray(realized_asset_returns, dtype=float)
    if realized_asset_returns.ndim != 2 or realized_asset_returns.shape[1] != target.size:
        raise ValueError("realized_asset_returns must have shape (days, assets)")
    gross_growth = np.prod(1.0 + realized_asset_returns, axis=0)
    end_values = target * gross_growth
    total = float(end_values.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("portfolio end value must be positive and finite")
    return end_values / total
