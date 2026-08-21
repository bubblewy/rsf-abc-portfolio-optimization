"""Canonical and risk-sensitive-foraging Artificial Bee Colony variants."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import numpy as np

from ..objective import OmegaObjective
from ..portfolio import repair_cardinality
from ..types import OptimizationResult


ABC_VARIANTS = {
    "standard_abc",
    "ht_abc",
    "rs_light_abc",
    "rsf_abc",
    "fixed_mix_abc",
}


@dataclass(frozen=True)
class ABCConfig:
    variant: str
    population_size: int = 30
    max_evaluations: int = 3000
    limit: int = 100
    kappa: float = 4.0
    student_df: float = 3.0
    student_clip: float = 4.0
    fixed_explore_probability: float = 0.5


class ABCOptimizer:
    def __init__(
        self,
        config: ABCConfig,
        cardinality: int,
        lower: float,
        upper: float,
    ) -> None:
        if config.variant not in ABC_VARIANTS:
            raise ValueError(f"Unknown ABC variant: {config.variant}")
        if config.population_size < 4 or config.max_evaluations < config.population_size:
            raise ValueError("invalid population or evaluation budget")
        if config.limit < 1 or config.student_df <= 0 or config.student_clip <= 0:
            raise ValueError("invalid ABC control parameter")
        if not 0.0 <= config.fixed_explore_probability <= 1.0:
            raise ValueError("fixed_explore_probability must be in [0, 1]")
        self.config = config
        self.cardinality = cardinality
        self.lower = lower
        self.upper = upper

    @staticmethod
    def _rank_probabilities(scores: np.ndarray) -> np.ndarray:
        n = scores.size
        order = np.argsort(-scores, kind="mergesort")
        rank_weights = np.arange(n, 0, -1, dtype=float)
        probabilities = np.empty(n, dtype=float)
        probabilities[order] = rank_weights / rank_weights.sum()
        return probabilities

    def _neighbor(
        self,
        source: np.ndarray,
        partner: np.ndarray,
        best: np.ndarray,
        budget: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, str, float | None]:
        raw = source.copy()
        coordinate = int(rng.integers(source.size))
        variant = self.config.variant

        if variant == "standard_abc":
            mode = "light_explore"
            p_explore = None
        elif variant == "ht_abc":
            mode = "heavy_explore"
            p_explore = 1.0
        elif variant == "fixed_mix_abc":
            p_explore = self.config.fixed_explore_probability
            mode = "heavy_explore" if rng.random() < p_explore else "conservative"
        else:
            argument = float(np.clip(self.config.kappa * budget, -700.0, 700.0))
            p_explore = 1.0 / (1.0 + math.exp(argument))
            if rng.random() < p_explore:
                mode = "heavy_explore" if variant == "rsf_abc" else "light_explore"
            else:
                mode = "conservative"

        if mode == "light_explore":
            multiplier = rng.uniform(-1.0, 1.0)
            raw[coordinate] = source[coordinate] + multiplier * (
                source[coordinate] - partner[coordinate]
            )
        elif mode == "heavy_explore":
            multiplier = float(
                np.clip(
                    rng.standard_t(self.config.student_df),
                    -self.config.student_clip,
                    self.config.student_clip,
                )
            )
            raw[coordinate] = source[coordinate] + multiplier * (
                source[coordinate] - partner[coordinate]
            )
        else:
            rho = rng.uniform(0.0, 0.5)
            raw[coordinate] = source[coordinate] + rho * (
                best[coordinate] - source[coordinate]
            )

        return (
            repair_cardinality(raw, self.cardinality, self.lower, self.upper),
            mode,
            p_explore,
        )

    @staticmethod
    def _distribution_summary(values: list[float]) -> dict[str, float | int] | None:
        if not values:
            return None
        array = np.asarray(values, dtype=float)
        q25, median, q75 = np.quantile(array, [0.25, 0.50, 0.75])
        return {
            "count": int(array.size),
            "mean": float(array.mean()),
            "std": float(array.std(ddof=0)),
            "min": float(array.min()),
            "q25": float(q25),
            "median": float(median),
            "q75": float(q75),
            "max": float(array.max()),
        }

    @staticmethod
    def _is_better(
        score: float,
        turnover: float,
        incumbent_score: float,
        incumbent_turnover: float,
    ) -> bool:
        return bool(
            score > incumbent_score + 1e-12
            or (
                abs(score - incumbent_score) <= 1e-12
                and turnover < incumbent_turnover - 1e-12
            )
        )

    def optimize(
        self,
        objective: OmegaObjective,
        initial_population: np.ndarray,
        rng: np.random.Generator,
    ) -> OptimizationResult:
        start = time.perf_counter()
        population = np.asarray(initial_population, dtype=float).copy()
        if population.shape != (self.config.population_size, objective.returns.shape[1]):
            raise ValueError("initial population shape does not match ABC configuration")

        scores, budgets, turnovers = objective.score_population(population)
        evaluations = self.config.population_size
        trials = np.zeros(self.config.population_size, dtype=int)
        best_index = int(np.argmax(scores))
        best_weights = population[best_index].copy()
        best_score = float(scores[best_index])
        best_turnover = float(turnovers[best_index])
        convergence_evaluations = [evaluations]
        convergence_best = [best_score]
        mode_counts = {"light_explore": 0, "heavy_explore": 0, "conservative": 0}
        proposal_budgets: list[float] = []
        proposal_probabilities: list[float] = []
        scouts = 0

        def attempt(source_index: int) -> None:
            nonlocal evaluations, best_weights, best_score, best_turnover
            partner_index = int(rng.integers(self.config.population_size - 1))
            if partner_index >= source_index:
                partner_index += 1
            proposal_budget = float(budgets[source_index])
            candidate, mode, p_explore = self._neighbor(
                population[source_index],
                population[partner_index],
                best_weights,
                proposal_budget,
                rng,
            )
            mode_counts[mode] += 1
            proposal_budgets.append(proposal_budget)
            if p_explore is not None:
                proposal_probabilities.append(float(p_explore))
            stats = objective.evaluate(candidate)
            evaluations += 1
            if self._is_better(
                stats.omega,
                stats.turnover,
                float(scores[source_index]),
                float(turnovers[source_index]),
            ):
                population[source_index] = candidate
                scores[source_index] = stats.omega
                budgets[source_index] = stats.budget
                turnovers[source_index] = stats.turnover
                trials[source_index] = 0
                if self._is_better(
                    stats.omega, stats.turnover, best_score, best_turnover
                ):
                    best_weights = candidate.copy()
                    best_score = float(stats.omega)
                    best_turnover = float(stats.turnover)
            else:
                trials[source_index] += 1

        while evaluations < self.config.max_evaluations:
            for source_index in range(self.config.population_size):
                if evaluations >= self.config.max_evaluations:
                    break
                attempt(source_index)

            if evaluations >= self.config.max_evaluations:
                convergence_evaluations.append(evaluations)
                convergence_best.append(best_score)
                break

            probabilities = self._rank_probabilities(scores)
            for _ in range(self.config.population_size):
                if evaluations >= self.config.max_evaluations:
                    break
                source_index = int(rng.choice(self.config.population_size, p=probabilities))
                attempt(source_index)

            if evaluations < self.config.max_evaluations:
                exhausted = np.flatnonzero(trials >= self.config.limit)
                for source_index in exhausted:
                    if evaluations >= self.config.max_evaluations:
                        break
                    raw = rng.normal(size=population.shape[1])
                    candidate = repair_cardinality(
                        raw, self.cardinality, self.lower, self.upper
                    )
                    stats = objective.evaluate(candidate)
                    evaluations += 1
                    scouts += 1
                    population[source_index] = candidate
                    scores[source_index] = stats.omega
                    budgets[source_index] = stats.budget
                    turnovers[source_index] = stats.turnover
                    trials[source_index] = 0
                    if self._is_better(
                        stats.omega, stats.turnover, best_score, best_turnover
                    ):
                        best_weights = candidate.copy()
                        best_score = float(stats.omega)
                        best_turnover = float(stats.turnover)

            convergence_evaluations.append(evaluations)
            convergence_best.append(best_score)

        runtime = time.perf_counter() - start
        total_modes = max(sum(mode_counts.values()), 1)
        budget_probability_correlation = None
        if len(proposal_probabilities) == len(proposal_budgets) and len(proposal_budgets) > 1:
            budget_array = np.asarray(proposal_budgets, dtype=float)
            probability_array = np.asarray(proposal_probabilities, dtype=float)
            if budget_array.std() > 1e-15 and probability_array.std() > 1e-15:
                budget_probability_correlation = float(
                    np.corrcoef(budget_array, probability_array)[0, 1]
                )
        diagnostics = {
            "variant": self.config.variant,
            "mode_counts": mode_counts,
            "mode_rates": {key: value / total_modes for key, value in mode_counts.items()},
            "proposal_budget_summary": self._distribution_summary(proposal_budgets),
            "proposal_p_explore_summary": self._distribution_summary(
                proposal_probabilities
            ),
            "budget_p_explore_correlation": budget_probability_correlation,
            "fixed_explore_probability": (
                float(self.config.fixed_explore_probability)
                if self.config.variant == "fixed_mix_abc"
                else None
            ),
            "scout_events": scouts,
            "best_turnover": best_turnover,
        }
        return OptimizationResult(
            method=self.config.variant,
            weights=best_weights,
            objective=best_score,
            evaluations=evaluations,
            runtime_seconds=runtime,
            convergence_evaluations=convergence_evaluations,
            convergence_best=convergence_best,
            diagnostics=diagnostics,
        )
