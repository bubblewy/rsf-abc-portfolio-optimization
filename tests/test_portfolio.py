import numpy as np

from rsfabc_portfolio.portfolio import (
    bounded_simplex_projection,
    drift_weights,
    is_feasible,
    repair_cardinality,
    turnover_one_way,
)


def test_bounded_simplex_projection_satisfies_constraints():
    values = np.array([-2.0, 0.1, 4.0, 1.5, 0.2])
    projected = bounded_simplex_projection(values, 0.02, 0.30)
    assert np.isclose(projected.sum(), 1.0, atol=1e-10)
    assert np.all(projected >= 0.02 - 1e-10)
    assert np.all(projected <= 0.30 + 1e-10)


def test_repair_has_exact_cardinality_and_stable_ties():
    scores = np.array([1.0, 1.0, 1.0, 0.5, 0.4, 0.3])
    weights = repair_cardinality(scores, 4, 0.05, 0.40)
    assert is_feasible(weights, 4, 0.05, 0.40)
    assert set(np.flatnonzero(weights > 1e-8)) == {0, 1, 2, 3}


def test_turnover_initial_cash_and_regular_rebalance():
    target = np.array([0.6, 0.4, 0.0])
    assert np.isclose(turnover_one_way(target, np.zeros(3), cash_pre=1.0), 1.0)
    pretrade = np.array([0.5, 0.5, 0.0])
    assert np.isclose(turnover_one_way(target, pretrade, cash_pre=0.0), 0.1)


def test_drift_weights_matches_buy_and_hold_values():
    target = np.array([0.5, 0.5])
    realized = np.array([[0.10, 0.0], [0.0, -0.10]])
    drifted = drift_weights(target, realized)
    expected = np.array([0.55, 0.45]) / 1.0
    assert np.allclose(drifted, expected)
