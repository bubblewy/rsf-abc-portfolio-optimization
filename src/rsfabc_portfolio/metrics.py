"""Financial path metrics used by the frozen analysis."""

from __future__ import annotations

import numpy as np


def performance_metrics(net_returns: np.ndarray, annualization: int = 252) -> dict[str, float]:
    returns = np.asarray(net_returns, dtype=float)
    if returns.ndim != 1 or returns.size == 0 or not np.isfinite(returns).all():
        raise ValueError("net_returns must be a finite one-dimensional array")
    wealth = np.cumprod(1.0 + returns)
    years = returns.size / annualization
    annual_return = float(wealth[-1] ** (1.0 / years) - 1.0) if years > 0 else np.nan
    volatility = float(returns.std(ddof=1) * np.sqrt(annualization))
    mean_annual = float(returns.mean() * annualization)
    sharpe = mean_annual / volatility if volatility > 0 else np.nan
    downside = np.minimum(returns, 0.0)
    downside_deviation = float(np.sqrt(np.mean(downside**2)) * np.sqrt(annualization))
    sortino = mean_annual / downside_deviation if downside_deviation > 0 else np.nan
    # Include initial wealth so an immediate loss is a drawdown from 1.0.
    running_peak = np.maximum.accumulate(np.concatenate(([1.0], wealth)))[1:]
    max_drawdown = float(np.min(wealth / running_peak - 1.0))
    losses = -returns
    var95 = float(np.quantile(losses, 0.95))
    tail = losses[losses >= var95]
    cvar95 = float(tail.mean()) if tail.size else var95
    gains = np.maximum(returns, 0.0).mean()
    shortfalls = np.maximum(-returns, 0.0).mean()
    omega = float(gains / (shortfalls + 1e-12))
    return {
        "observations": float(returns.size),
        "cumulative_return": float(wealth[-1] - 1.0),
        "annualized_return": annual_return,
        "annualized_volatility": volatility,
        "sharpe_zero_rf": float(sharpe),
        "sortino_zero_threshold": float(sortino),
        "max_drawdown": max_drawdown,
        "cvar95_daily_loss": cvar95,
        "omega_zero_threshold": omega,
        "terminal_wealth": float(wealth[-1]),
    }
