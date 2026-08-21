"""Budget-matched DE/rand/1/bin baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..objective import OmegaObjective
from ..portfolio import repair_cardinality
from ..types import OptimizationResult


@dataclass(frozen=True)
class DEConfig:
    population_size: int = 30
    max_evaluations: int = 3000
    differential_weight: float = 0.5
    crossover_rate: float = 0.9


class DifferentialEvolutionOptimizer:
    def __init__(self, config: DEConfig, cardinality: int, lower: float, upper: float):
        if config.population_size < 4 or config.max_evaluations < config.population_size:
            raise ValueError("invalid population or evaluation budget")
        if not 0 < config.differential_weight <= 2 or not 0 <= config.crossover_rate <= 1:
            raise ValueError("invalid DE control parameter")
        self.config = config
        self.cardinality = cardinality
        self.lower = lower
        self.upper = upper

    def optimize(
        self,
        objective: OmegaObjective,
        initial_population: np.ndarray,
        rng: np.random.Generator,
    ) -> OptimizationResult:
        start = time.perf_counter()
        population = np.asarray(initial_population, dtype=float).copy()
        expected_shape = (self.config.population_size, objective.returns.shape[1])
        if population.shape != expected_shape:
            raise ValueError("initial population shape does not match DE configuration")
        scores, _, turnovers = objective.score_population(population)
        evaluations = self.config.population_size
        best_index = int(np.argmax(scores))
        best_weights = population[best_index].copy()
        best_score = float(scores[best_index])
        best_turnover = float(turnovers[best_index])
        convergence_evaluations = [evaluations]
        convergence_best = [best_score]
        generation = 0

        while evaluations < self.config.max_evaluations:
            candidate_count = min(
                self.config.population_size,
                self.config.max_evaluations - evaluations,
            )
            trials = []
            for i in range(candidate_count):
                pool = np.delete(np.arange(self.config.population_size), i)
                r1, r2, r3 = rng.choice(pool, size=3, replace=False)
                mutant = population[r1] + self.config.differential_weight * (
                    population[r2] - population[r3]
                )
                mask = rng.random(population.shape[1]) < self.config.crossover_rate
                mask[int(rng.integers(population.shape[1]))] = True
                raw_trial = np.where(mask, mutant, population[i])
                trials.append(
                    repair_cardinality(
                        raw_trial, self.cardinality, self.lower, self.upper
                    )
                )
            trial_population = np.vstack(trials)
            trial_scores, _, trial_turnovers = objective.score_population(trial_population)
            evaluations += candidate_count
            for i in range(candidate_count):
                better = bool(
                    trial_scores[i] > scores[i] + 1e-12
                    or (
                        abs(trial_scores[i] - scores[i]) <= 1e-12
                        and trial_turnovers[i] < turnovers[i] - 1e-12
                    )
                )
                if better:
                    population[i] = trial_population[i]
                    scores[i] = trial_scores[i]
                    turnovers[i] = trial_turnovers[i]
                    if bool(
                        trial_scores[i] > best_score + 1e-12
                        or (
                            abs(trial_scores[i] - best_score) <= 1e-12
                            and trial_turnovers[i] < best_turnover - 1e-12
                        )
                    ):
                        best_weights = trial_population[i].copy()
                        best_score = float(trial_scores[i])
                        best_turnover = float(trial_turnovers[i])
            generation += 1
            convergence_evaluations.append(evaluations)
            convergence_best.append(best_score)

        return OptimizationResult(
            method="de",
            weights=best_weights,
            objective=best_score,
            evaluations=evaluations,
            runtime_seconds=time.perf_counter() - start,
            convergence_evaluations=convergence_evaluations,
            convergence_best=convergence_best,
            diagnostics={"best_turnover": best_turnover, "generations": generation},
        )
