"""Budget-matched Particle Swarm Optimization baseline."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from ..objective import OmegaObjective
from ..portfolio import repair_cardinality
from ..types import OptimizationResult


@dataclass(frozen=True)
class PSOConfig:
    population_size: int = 30
    max_evaluations: int = 3000
    inertia_start: float = 0.9
    inertia_end: float = 0.4
    c1: float = 1.49618
    c2: float = 1.49618


class ParticleSwarmOptimizer:
    def __init__(self, config: PSOConfig, cardinality: int, lower: float, upper: float):
        if config.population_size < 4 or config.max_evaluations < config.population_size:
            raise ValueError("invalid population or evaluation budget")
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
        positions = np.asarray(initial_population, dtype=float).copy()
        expected_shape = (self.config.population_size, objective.returns.shape[1])
        if positions.shape != expected_shape:
            raise ValueError("initial population shape does not match PSO configuration")
        velocities = rng.normal(0.0, 0.05, size=positions.shape)
        scores, _, turnovers = objective.score_population(positions)
        evaluations = self.config.population_size
        personal_positions = positions.copy()
        personal_scores = scores.copy()
        personal_turnovers = turnovers.copy()
        best_index = int(np.argmax(personal_scores))
        global_position = personal_positions[best_index].copy()
        global_score = float(personal_scores[best_index])
        global_turnover = float(personal_turnovers[best_index])
        convergence_evaluations = [evaluations]
        convergence_best = [global_score]

        generations = max(
            1,
            (self.config.max_evaluations - self.config.population_size)
            // self.config.population_size,
        )
        generation = 0
        while evaluations < self.config.max_evaluations:
            progress = min(generation / max(generations - 1, 1), 1.0)
            inertia = self.config.inertia_start + progress * (
                self.config.inertia_end - self.config.inertia_start
            )
            r1 = rng.random(size=positions.shape)
            r2 = rng.random(size=positions.shape)
            velocities = (
                inertia * velocities
                + self.config.c1 * r1 * (personal_positions - positions)
                + self.config.c2 * r2 * (global_position[None, :] - positions)
            )
            raw_positions = positions + velocities
            candidate_count = min(
                self.config.population_size,
                self.config.max_evaluations - evaluations,
            )
            candidates = np.vstack(
                [
                    repair_cardinality(
                        raw_positions[i], self.cardinality, self.lower, self.upper
                    )
                    for i in range(candidate_count)
                ]
            )
            candidate_scores, _, candidate_turnovers = objective.score_population(candidates)
            evaluations += candidate_count
            for i in range(candidate_count):
                positions[i] = candidates[i]
                better = bool(
                    candidate_scores[i] > personal_scores[i] + 1e-12
                    or (
                        abs(candidate_scores[i] - personal_scores[i]) <= 1e-12
                        and candidate_turnovers[i] < personal_turnovers[i] - 1e-12
                    )
                )
                if better:
                    personal_positions[i] = candidates[i]
                    personal_scores[i] = candidate_scores[i]
                    personal_turnovers[i] = candidate_turnovers[i]
                    if bool(
                        candidate_scores[i] > global_score + 1e-12
                        or (
                            abs(candidate_scores[i] - global_score) <= 1e-12
                            and candidate_turnovers[i] < global_turnover - 1e-12
                        )
                    ):
                        global_position = candidates[i].copy()
                        global_score = float(candidate_scores[i])
                        global_turnover = float(candidate_turnovers[i])
            generation += 1
            convergence_evaluations.append(evaluations)
            convergence_best.append(global_score)

        return OptimizationResult(
            method="pso",
            weights=global_position,
            objective=global_score,
            evaluations=evaluations,
            runtime_seconds=time.perf_counter() - start,
            convergence_evaluations=convergence_evaluations,
            convergence_best=convergence_best,
            diagnostics={"best_turnover": global_turnover, "generations": generation},
        )
