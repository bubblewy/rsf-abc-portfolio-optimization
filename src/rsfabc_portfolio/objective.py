"""Cost-aware empirical Omega objective and budget signal."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .portfolio import turnover_one_way


@dataclass(frozen=True)
class OmegaStats:
    omega: float
    budget: float
    gain: float
    loss: float
    mean_excess: float
    turnover: float
    cost_shift: float


class OmegaObjective:
    def __init__(
        self,
        returns: np.ndarray,
        pretrade: np.ndarray,
        cash_pre: float,
        cost_rate: float,
        amortization_days: int,
        threshold: float,
        epsilon: float = 1e-12,
    ) -> None:
        self.returns = np.asarray(returns, dtype=float)
        self.pretrade = np.asarray(pretrade, dtype=float)
        if self.returns.ndim != 2:
            raise ValueError("returns must have shape (scenarios, assets)")
        if self.pretrade.shape != (self.returns.shape[1],):
            raise ValueError("pretrade must match the asset dimension")
        if amortization_days <= 0 or cost_rate < 0 or epsilon <= 0:
            raise ValueError("invalid objective parameters")
        self.cash_pre = float(cash_pre)
        self.cost_rate = float(cost_rate)
        self.amortization_days = int(amortization_days)
        self.threshold = float(threshold)
        self.epsilon = float(epsilon)

    def evaluate(self, weights: np.ndarray) -> OmegaStats:
        weights = np.asarray(weights, dtype=float)
        turnover = turnover_one_way(weights, self.pretrade, self.cash_pre)
        cost_shift = self.cost_rate * turnover / self.amortization_days
        excess = self.returns @ weights - cost_shift - self.threshold
        gain = float(np.maximum(excess, 0.0).mean())
        loss = float(np.maximum(-excess, 0.0).mean())
        omega = gain / (loss + self.epsilon)
        budget = (gain - loss) / (gain + loss + self.epsilon)
        return OmegaStats(
            omega=omega,
            budget=budget,
            gain=gain,
            loss=loss,
            mean_excess=float(excess.mean()),
            turnover=turnover,
            cost_shift=cost_shift,
        )

    def score(self, weights: np.ndarray) -> float:
        return self.evaluate(weights).omega

    def score_population(self, population: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        population = np.asarray(population, dtype=float)
        if population.ndim != 2 or population.shape[1] != self.returns.shape[1]:
            raise ValueError("population must have shape (candidates, assets)")
        turnovers = 0.5 * (
            np.abs(population - self.pretrade[None, :]).sum(axis=1) + abs(self.cash_pre)
        )
        shifts = self.cost_rate * turnovers / self.amortization_days
        excess = self.returns @ population.T - shifts[None, :] - self.threshold
        gains = np.maximum(excess, 0.0).mean(axis=0)
        losses = np.maximum(-excess, 0.0).mean(axis=0)
        omegas = gains / (losses + self.epsilon)
        budgets = (gains - losses) / (gains + losses + self.epsilon)
        return omegas, budgets, turnovers
