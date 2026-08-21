"""Method dispatch, walk-forward execution, and runtime-only pilot."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .algorithms import ABCOptimizer, DifferentialEvolutionOptimizer, ParticleSwarmOptimizer
from .algorithms.abc import ABCConfig
from .algorithms.de import DEConfig
from .algorithms.pso import PSOConfig
from .baselines import equal_weight_k, equal_weight_n, minimum_variance_k
from .data import sha256_file
from .milp import OmegaMILPConfig, solve_omega_milp
from .objective import OmegaObjective
from .portfolio import drift_weights, random_feasible_population, turnover_one_way
from .provenance import environment_record, utc_run_id, write_json
from .schedule import build_monthly_schedule
from .types import OptimizationResult


STOCHASTIC_METHODS = {
    "rsf_abc",
    "standard_abc",
    "ht_abc",
    "rs_light_abc",
    "fixed_mix_abc",
    "pso",
    "de",
}
DETERMINISTIC_METHODS = {"ew_n", "ew_k", "mv_k", "omega_milp"}
ALL_METHODS = STOCHASTIC_METHODS | DETERMINISTIC_METHODS


def make_objective(
    train_returns: np.ndarray,
    pretrade: np.ndarray,
    cash_pre: float,
    portfolio_config: dict[str, Any],
    threshold: float | None = None,
) -> OmegaObjective:
    return OmegaObjective(
        returns=train_returns,
        pretrade=pretrade,
        cash_pre=cash_pre,
        cost_rate=float(portfolio_config["transaction_cost_bps"]) / 10_000.0,
        amortization_days=int(portfolio_config["cost_amortization_days"]),
        threshold=(
            float(portfolio_config["omega_threshold"])
            if threshold is None
            else float(threshold)
        ),
        epsilon=float(portfolio_config["epsilon"]),
    )


def run_method(
    method: str,
    objective: OmegaObjective,
    portfolio_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    seed: int,
    initial_population: np.ndarray | None = None,
    fallback_weights: np.ndarray | None = None,
) -> OptimizationResult:
    if method not in ALL_METHODS:
        raise ValueError(f"Unknown method: {method}")
    cardinality = int(portfolio_config["cardinality"])
    lower = float(portfolio_config["lower_bound"])
    upper = float(portfolio_config["upper_bound"])

    if method in STOCHASTIC_METHODS:
        if initial_population is None:
            init_rng = np.random.default_rng(seed)
            initial_population = random_feasible_population(
                init_rng,
                int(optimizer_config["population_size"]),
                objective.returns.shape[1],
                cardinality,
                lower,
                upper,
            )
        rng = np.random.default_rng(seed)
        if method.endswith("_abc"):
            optimizer = ABCOptimizer(
                ABCConfig(
                    variant=method,
                    population_size=int(optimizer_config["population_size"]),
                    max_evaluations=int(optimizer_config["max_evaluations"]),
                    limit=int(optimizer_config["abc_limit"]),
                    kappa=float(optimizer_config["kappa"]),
                    student_df=float(optimizer_config["student_df"]),
                    student_clip=float(optimizer_config["student_clip"]),
                    fixed_explore_probability=float(
                        optimizer_config.get("fixed_explore_probability", 0.5)
                    ),
                ),
                cardinality,
                lower,
                upper,
            )
        elif method == "pso":
            optimizer = ParticleSwarmOptimizer(
                PSOConfig(
                    population_size=int(optimizer_config["population_size"]),
                    max_evaluations=int(optimizer_config["max_evaluations"]),
                    inertia_start=float(optimizer_config["pso_inertia_start"]),
                    inertia_end=float(optimizer_config["pso_inertia_end"]),
                    c1=float(optimizer_config["pso_c1"]),
                    c2=float(optimizer_config["pso_c2"]),
                ),
                cardinality,
                lower,
                upper,
            )
        else:
            optimizer = DifferentialEvolutionOptimizer(
                DEConfig(
                    population_size=int(optimizer_config["population_size"]),
                    max_evaluations=int(optimizer_config["max_evaluations"]),
                    differential_weight=float(optimizer_config["de_f"]),
                    crossover_rate=float(optimizer_config["de_cr"]),
                ),
                cardinality,
                lower,
                upper,
            )
        return optimizer.optimize(objective, initial_population.copy(), rng)

    if method == "ew_n":
        return equal_weight_n(objective)
    if method == "ew_k":
        return equal_weight_k(objective, cardinality, lower, upper)
    if method == "mv_k":
        return minimum_variance_k(objective, cardinality, lower, upper)
    return solve_omega_milp(
        objective,
        OmegaMILPConfig(
            cardinality=cardinality,
            lower=lower,
            upper=upper,
        ),
        fallback_weights=fallback_weights,
    )


def run_walk_forward_path(
    frame: pd.DataFrame,
    method: str,
    seed: int,
    portfolio_config: dict[str, Any],
    optimizer_config: dict[str, Any],
    train_days: int,
) -> dict[str, Any]:
    windows = build_monthly_schedule(frame.index, train_days)
    n_assets = frame.shape[1]
    cardinality = int(portfolio_config["cardinality"])
    lower = float(portfolio_config["lower_bound"])
    upper = float(portfolio_config["upper_bound"])
    cost_rate = float(portfolio_config["transaction_cost_bps"]) / 10_000.0
    pretrade = np.zeros(n_assets)
    cash_pre = 1.0
    prior_target: np.ndarray | None = None
    net_returns: list[float] = []
    daily_dates: list[str] = []
    rebalance_records: list[dict[str, Any]] = []

    for window in windows:
        train = frame.iloc[window.train_positions].to_numpy()
        hold = frame.iloc[window.hold_positions].to_numpy()
        threshold_mode = portfolio_config.get("omega_threshold_mode", "fixed")
        if threshold_mode == "fixed":
            threshold = float(portfolio_config["omega_threshold"])
        elif threshold_mode == "equal_weight_mean":
            threshold = float(train.mean(axis=1).mean())
        else:
            raise ValueError(f"Unknown omega_threshold_mode: {threshold_mode}")
        objective = make_objective(
            train, pretrade, cash_pre, portfolio_config, threshold=threshold
        )
        initial_population = None
        window_seed = int(np.random.SeedSequence([seed, window.index, 2909]).generate_state(1)[0])
        if method in STOCHASTIC_METHODS:
            init_rng = np.random.default_rng(
                int(np.random.SeedSequence([seed, window.index, 1709]).generate_state(1)[0])
            )
            initial_population = random_feasible_population(
                init_rng,
                int(optimizer_config["population_size"]),
                n_assets,
                cardinality,
                lower,
                upper,
            )
        result = run_method(
            method,
            objective,
            portfolio_config,
            optimizer_config,
            window_seed,
            initial_population,
            fallback_weights=prior_target,
        )
        turnover = turnover_one_way(result.weights, pretrade, cash_pre)
        gross = hold @ result.weights
        net = gross.copy()
        net[0] = (1.0 - cost_rate * turnover) * (1.0 + gross[0]) - 1.0
        net_returns.extend(net.tolist())
        daily_dates.extend(frame.index[window.hold_positions].strftime("%Y-%m-%d").tolist())
        rebalance_records.append(
            {
                "window_index": window.index,
                "train_start": window.train_start,
                "train_end": window.train_end,
                "hold_start": window.hold_start,
                "hold_end": window.hold_end,
                "seed": seed,
                "window_seed": window_seed,
                "omega_threshold": threshold,
                "omega_threshold_mode": threshold_mode,
                "turnover": float(turnover),
                "transaction_cost": float(cost_rate * turnover),
                "optimizer": result.to_record(),
            }
        )
        pretrade = drift_weights(result.weights, hold)
        cash_pre = 0.0
        prior_target = result.weights.copy()

    return {
        "method": method,
        "seed": seed,
        "daily_dates": daily_dates,
        "net_returns": net_returns,
        "rebalances": rebalance_records,
    }


def run_runtime_pilot(
    frame: pd.DataFrame,
    config: dict[str, Any],
    data_path: Path,
    output_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    portfolio_config = config["portfolio"]
    optimizer_config = config["optimizers"]
    train_days = int(config["walk_forward"]["train_days"])
    windows = build_monthly_schedule(frame.index, train_days)
    window = windows[0]
    train = frame.iloc[window.train_positions].to_numpy()
    pretrade = np.zeros(frame.shape[1])
    objective = make_objective(train, pretrade, 1.0, portfolio_config)
    seed = int(optimizer_config["seed"])
    init_rng = np.random.default_rng(seed)
    initial_population = random_feasible_population(
        init_rng,
        int(optimizer_config["population_size"]),
        frame.shape[1],
        int(portfolio_config["cardinality"]),
        float(portfolio_config["lower_bound"]),
        float(portfolio_config["upper_bound"]),
    )

    records = []
    methods = list(optimizer_config["methods"])
    for method in methods:
        result = run_method(
            method,
            objective,
            portfolio_config,
            optimizer_config,
            seed,
            initial_population,
        )
        record = result.to_record()
        record.pop("weights")
        records.append(record)

    times = {record["method"]: float(record["runtime_seconds"]) for record in records}
    window_count = len(windows)
    main_cpu_seconds = sum(times.values()) * int(config["pilot"]["full_path_seeds"]) * window_count
    diagnostics_cpu_seconds = (
        sum(times.values())
        * int(config["pilot"]["diagnostic_dates"])
        * int(config["pilot"]["diagnostic_seeds"])
    )
    reduced_methods = ["rsf_abc", "standard_abc", "de"]
    robustness_conditions = 7
    robustness_cpu_seconds = (
        sum(times[name] for name in reduced_methods)
        * int(config["pilot"]["robustness_stochastic_seeds"])
        * window_count
        * robustness_conditions
    )
    total_cpu_seconds = main_cpu_seconds + diagnostics_cpu_seconds + robustness_cpu_seconds
    logical_cpus = os.cpu_count() or 1
    local_workers = max(1, min(6, logical_cpus - 2 if logical_cpus > 2 else 1))
    cloud_workers = 24
    local_wall_seconds = total_cpu_seconds / local_workers * 1.25
    cloud_wall_seconds = total_cpu_seconds / cloud_workers * 1.20
    saved_seconds = local_wall_seconds - cloud_wall_seconds
    cloud_material = local_wall_seconds >= 3600 and saved_seconds >= 1800

    payload = {
        "run_id": utc_run_id("pilot"),
        "purpose": "runtime-only; no out-of-sample performance inspected",
        "protocol_version": config["project"]["protocol_version"],
        "data": {
            "path": str(data_path),
            "sha256": sha256_file(data_path),
            "rows": len(frame),
            "assets": frame.shape[1],
        },
        "window": {
            "train_start": window.train_start,
            "train_end": window.train_end,
            "hold_start_not_evaluated": window.hold_start,
            "training_observations": len(window.train_positions),
        },
        "optimizer_records": records,
        "projection": {
            "walk_forward_windows": window_count,
            "main_cpu_hours": main_cpu_seconds / 3600.0,
            "diagnostics_cpu_hours": diagnostics_cpu_seconds / 3600.0,
            "robustness_cpu_hours": robustness_cpu_seconds / 3600.0,
            "total_stochastic_cpu_hours": total_cpu_seconds / 3600.0,
            "local_workers_assumed": local_workers,
            "local_wall_hours": local_wall_seconds / 3600.0,
            "cloud_workers_assumed": cloud_workers,
            "cloud_wall_hours": cloud_wall_seconds / 3600.0,
            "estimated_hours_saved": saved_seconds / 3600.0,
            "cloud_cpu_material": cloud_material,
            "gpu_recommended": False,
            "notes": "Projection excludes deterministic MILP time and is intentionally conservative for smaller universes.",
        },
        "environment": environment_record(),
    }
    output_path = Path(output_dir) / f"{payload['run_id']}.json"
    write_json(output_path, payload)
    return output_path, payload
