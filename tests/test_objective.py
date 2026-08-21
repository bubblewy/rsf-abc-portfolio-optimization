import numpy as np

from rsfabc_portfolio.objective import OmegaObjective


def test_gain_loss_identity_and_budget_bounds():
    returns = np.array(
        [
            [0.01, -0.02],
            [-0.01, 0.03],
            [0.02, 0.01],
            [-0.02, -0.01],
        ]
    )
    objective = OmegaObjective(
        returns,
        pretrade=np.array([0.5, 0.5]),
        cash_pre=0.0,
        cost_rate=0.0,
        amortization_days=21,
        threshold=0.0,
    )
    stats = objective.evaluate(np.array([0.6, 0.4]))
    assert np.isclose(stats.gain - stats.loss, stats.mean_excess)
    assert -1.0 <= stats.budget <= 1.0
    assert stats.omega >= 0.0


def test_population_scores_equal_scalar_scores():
    rng = np.random.default_rng(4)
    returns = rng.normal(0.0002, 0.01, size=(50, 4))
    objective = OmegaObjective(
        returns,
        pretrade=np.full(4, 0.25),
        cash_pre=0.0,
        cost_rate=0.0025,
        amortization_days=21,
        threshold=0.0,
    )
    population = np.array(
        [[0.4, 0.3, 0.2, 0.1], [0.1, 0.2, 0.3, 0.4]], dtype=float
    )
    scores, budgets, turnovers = objective.score_population(population)
    for i, weights in enumerate(population):
        scalar = objective.evaluate(weights)
        assert np.isclose(scores[i], scalar.omega)
        assert np.isclose(budgets[i], scalar.budget)
        assert np.isclose(turnovers[i], scalar.turnover)


def test_cost_shift_uses_one_way_turnover():
    returns = np.zeros((10, 2))
    objective = OmegaObjective(
        returns,
        pretrade=np.zeros(2),
        cash_pre=1.0,
        cost_rate=0.0025,
        amortization_days=20,
        threshold=0.0,
    )
    stats = objective.evaluate(np.array([0.5, 0.5]))
    assert np.isclose(stats.turnover, 1.0)
    assert np.isclose(stats.cost_shift, 0.0025 / 20)
