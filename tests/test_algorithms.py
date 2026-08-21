import numpy as np
import pytest

from rsfabc_portfolio.objective import OmegaObjective
from rsfabc_portfolio.portfolio import is_feasible, random_feasible_population
from rsfabc_portfolio.runner import run_method


@pytest.fixture
def small_problem():
    rng = np.random.default_rng(9)
    returns = rng.normal(0.0003, 0.012, size=(120, 8))
    objective = OmegaObjective(
        returns,
        pretrade=np.zeros(8),
        cash_pre=1.0,
        cost_rate=0.0025,
        amortization_days=21,
        threshold=0.0,
    )
    portfolio = {
        "cardinality": 4,
        "lower_bound": 0.05,
        "upper_bound": 0.50,
        "transaction_cost_bps": 25,
        "cost_amortization_days": 21,
        "omega_threshold": 0.0,
        "epsilon": 1e-12,
    }
    optimizers = {
        "population_size": 10,
        "max_evaluations": 100,
        "abc_limit": 20,
        "kappa": 4.0,
        "student_df": 3.0,
        "student_clip": 4.0,
        "fixed_explore_probability": 0.35,
        "pso_inertia_start": 0.9,
        "pso_inertia_end": 0.4,
        "pso_c1": 1.49618,
        "pso_c2": 1.49618,
        "de_f": 0.5,
        "de_cr": 0.9,
    }
    initial = random_feasible_population(
        np.random.default_rng(123), 10, 8, 4, 0.05, 0.50
    )
    return objective, portfolio, optimizers, initial


@pytest.mark.parametrize(
    "method",
    [
        "standard_abc",
        "ht_abc",
        "rs_light_abc",
        "rsf_abc",
        "fixed_mix_abc",
        "pso",
        "de",
    ],
)
def test_stochastic_methods_are_feasible_budget_matched_and_reproducible(
    small_problem, method
):
    objective, portfolio, optimizers, initial = small_problem
    first = run_method(method, objective, portfolio, optimizers, 777, initial)
    second = run_method(method, objective, portfolio, optimizers, 777, initial)
    assert first.evaluations == 100
    assert second.evaluations == 100
    assert np.allclose(first.weights, second.weights)
    assert np.isclose(first.objective, second.objective)
    assert is_feasible(first.weights, 4, 0.05, 0.50)
    assert first.objective >= objective.score(initial[np.argmax(objective.score_population(initial)[0])])


def test_state_and_branch_diagnostics_are_recorded(small_problem):
    objective, portfolio, optimizers, initial = small_problem
    fixed = run_method(
        "fixed_mix_abc", objective, portfolio, optimizers, 777, initial
    )
    fixed_diagnostics = fixed.diagnostics
    assert fixed_diagnostics["proposal_budget_summary"]["count"] > 0
    assert np.isclose(
        fixed_diagnostics["proposal_p_explore_summary"]["mean"], 0.35
    )
    assert fixed_diagnostics["budget_p_explore_correlation"] is None

    rsf = run_method("rsf_abc", objective, portfolio, optimizers, 777, initial)
    rsf_diagnostics = rsf.diagnostics
    assert 0.0 <= rsf_diagnostics["proposal_p_explore_summary"]["mean"] <= 1.0
    assert rsf_diagnostics["budget_p_explore_correlation"] < 0.0
