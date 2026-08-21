import numpy as np

from rsfabc_portfolio.baselines import equal_weight_k, minimum_variance_k
from rsfabc_portfolio.milp import (
    OmegaMILPConfig,
    _finite_float_or_none,
    solve_omega_milp,
)
from rsfabc_portfolio.objective import OmegaObjective
from rsfabc_portfolio.portfolio import is_feasible


def _problem():
    returns = np.array(
        [
            [0.020, 0.010, -0.010, 0.000],
            [0.010, 0.012, -0.020, 0.002],
            [-0.005, 0.006, 0.010, -0.010],
            [0.015, -0.004, 0.002, 0.001],
            [-0.010, 0.008, 0.005, -0.003],
            [0.012, 0.009, -0.004, 0.003],
        ]
    )
    return OmegaObjective(
        returns,
        pretrade=np.zeros(4),
        cash_pre=1.0,
        cost_rate=0.0,
        amortization_days=21,
        threshold=0.0,
    )


def _below_one_problem():
    returns = np.array(
        [
            [0.010, 0.008, 0.006, 0.004],
            [-0.030, -0.025, -0.020, -0.018],
            [-0.020, -0.018, -0.016, -0.014],
            [0.005, 0.004, 0.003, 0.002],
            [-0.012, -0.010, -0.011, -0.009],
            [0.004, 0.003, 0.002, 0.001],
        ]
    )
    return OmegaObjective(
        returns,
        pretrade=np.zeros(4),
        cash_pre=1.0,
        cost_rate=0.0,
        amortization_days=21,
        threshold=0.0,
    )


def test_deterministic_cardinality_baselines_are_feasible():
    objective = _problem()
    for result in (
        equal_weight_k(objective, 2, 0.2, 0.8),
        minimum_variance_k(objective, 2, 0.2, 0.8),
    ):
        assert is_feasible(result.weights, 2, 0.2, 0.8)
        assert np.isfinite(result.objective)


def test_omega_milp_returns_feasible_incumbent():
    objective = _problem()
    result = solve_omega_milp(
        objective,
        OmegaMILPConfig(
            cardinality=2,
            lower=0.2,
            upper=0.8,
            time_limit_seconds=5.0,
            max_iterations=15,
        ),
    )
    assert is_feasible(result.weights, 2, 0.2, 0.8)
    assert np.isfinite(result.objective)
    assert result.objective >= equal_weight_k(objective, 2, 0.2, 0.8).objective - 1e-6


def test_nonfinite_solver_gap_is_serialized_as_missing():
    assert _finite_float_or_none(None) is None
    assert _finite_float_or_none(np.inf) is None
    assert _finite_float_or_none(-np.inf) is None
    assert _finite_float_or_none(np.nan) is None
    assert _finite_float_or_none(0.125) == 0.125


def test_omega_milp_handles_optimum_below_one():
    objective = _below_one_problem()
    result = solve_omega_milp(
        objective,
        OmegaMILPConfig(
            cardinality=2,
            lower=0.2,
            upper=0.8,
            time_limit_seconds=10.0,
            max_iterations=20,
        ),
    )
    brute_best = -np.inf
    for left in range(4):
        for right in range(left + 1, 4):
            for left_weight in np.linspace(0.2, 0.8, 601):
                weights = np.zeros(4)
                weights[left] = left_weight
                weights[right] = 1.0 - left_weight
                brute_best = max(brute_best, objective.score(weights))
    assert result.success
    assert result.diagnostics["signed_fallback_used"] is True
    assert 0.0 < result.objective < 1.0
    assert result.objective >= brute_best - 1e-5
    assert is_feasible(result.weights, 2, 0.2, 0.8)
