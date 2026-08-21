import numpy as np
import pandas as pd

from rsfabc_portfolio.runner import run_walk_forward_path


def _configs(threshold_mode="fixed"):
    portfolio = {
        "cardinality": 4,
        "lower_bound": 0.05,
        "upper_bound": 0.50,
        "transaction_cost_bps": 25,
        "cost_amortization_days": 21,
        "omega_threshold": 0.0,
        "omega_threshold_mode": threshold_mode,
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
    return portfolio, optimizers


def test_walk_forward_path_is_complete_reproducible_and_cost_aware():
    rng = np.random.default_rng(12)
    index = pd.bdate_range("2018-01-02", periods=720)
    frame = pd.DataFrame(
        rng.normal(0.0002, 0.01, size=(len(index), 8)), index=index
    )
    portfolio, optimizers = _configs()
    first = run_walk_forward_path(
        frame, "standard_abc", 101, portfolio, optimizers, train_days=504
    )
    second = run_walk_forward_path(
        frame, "standard_abc", 101, portfolio, optimizers, train_days=504
    )
    assert first["net_returns"] == second["net_returns"]
    assert len(first["daily_dates"]) == len(first["net_returns"])
    assert len(first["rebalances"]) > 0
    assert np.isclose(first["rebalances"][0]["turnover"], 1.0)
    assert np.isclose(first["rebalances"][0]["transaction_cost"], 0.0025)
    assert all(
        pd.Timestamp(row["train_end"]) < pd.Timestamp(row["hold_start"])
        for row in first["rebalances"]
    )


def test_equal_weight_threshold_is_fitted_inside_each_training_window():
    rng = np.random.default_rng(13)
    index = pd.bdate_range("2018-01-02", periods=720)
    frame = pd.DataFrame(
        rng.normal(0.0004, 0.01, size=(len(index), 8)), index=index
    )
    portfolio, optimizers = _configs("equal_weight_mean")
    result = run_walk_forward_path(
        frame, "ew_k", 0, portfolio, optimizers, train_days=504
    )
    first = result["rebalances"][0]
    expected = frame.loc[first["train_start"] : first["train_end"]].to_numpy().mean()
    assert np.isclose(first["omega_threshold"], expected)
