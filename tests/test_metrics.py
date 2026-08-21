import numpy as np

from rsfabc_portfolio.metrics import performance_metrics


def test_metrics_are_finite_for_non_degenerate_path():
    returns = np.array([0.01, -0.005, 0.002, -0.003, 0.006] * 60)
    metrics = performance_metrics(returns)
    assert metrics["terminal_wealth"] > 0
    assert metrics["omega_zero_threshold"] > 0
    assert metrics["max_drawdown"] <= 0
    assert np.isfinite(metrics["annualized_volatility"])


def test_max_drawdown_includes_initial_wealth_peak():
    metrics = performance_metrics(np.array([-0.10, 0.05]))
    assert np.isclose(metrics["max_drawdown"], -0.10)
