"""Resumable confirmatory batch orchestration."""

from __future__ import annotations

import json
import time
import traceback
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from joblib import Parallel, delayed

from .config import project_path
from .data import load_processed_dataset, sha256_file
from .objective import OmegaObjective
from .portfolio import random_feasible_population
from .provenance import environment_record, write_json
from .runner import run_method, run_walk_forward_path
from .schedule import build_monthly_schedule, evenly_spaced_window_indices


def _dataset_path(project_root: Path, config: dict[str, Any], universe: int) -> Path:
    data = config["data"]
    return project_path(project_root, data["processed_dir"]) / (
        f"industry_{universe}_daily_{data['start_date']}_{data['end_date']}.csv.gz"
    )


def _condition_portfolio(config: dict[str, Any], condition: dict[str, Any]) -> dict[str, Any]:
    portfolio = deepcopy(config["portfolio_defaults"])
    for key in (
        "cardinality",
        "lower_bound",
        "upper_bound",
        "transaction_cost_bps",
        "cost_amortization_days",
        "omega_threshold",
        "omega_threshold_mode",
        "epsilon",
    ):
        if key in condition:
            portfolio[key] = condition[key]
    return portfolio


def _path_filename(condition_id: str, method: str, seed: int) -> str:
    return f"{condition_id}__{method}__seed{seed}.json"


def _run_path_task(
    data_path: Path,
    condition_id: str,
    method: str,
    seed: int,
    portfolio: dict[str, Any],
    optimizer_config: dict[str, Any],
    train_days: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        return {"status": "skipped_existing", "path": str(output_path)}
    start = time.perf_counter()
    try:
        frame = load_processed_dataset(data_path)
        payload = run_walk_forward_path(
            frame,
            method,
            seed,
            portfolio,
            optimizer_config,
            train_days,
        )
        payload.update(
            {
                "condition_id": condition_id,
                "universe": int(frame.shape[1]),
                "portfolio_config": portfolio,
                "optimizer_config": optimizer_config,
                "data_path": str(data_path),
                "data_sha256": sha256_file(data_path),
                "task_runtime_seconds": time.perf_counter() - start,
            }
        )
        write_json(output_path, payload)
        failures = sum(not row["optimizer"]["success"] for row in payload["rebalances"])
        return {
            "status": "completed",
            "path": str(output_path),
            "runtime_seconds": payload["task_runtime_seconds"],
            "optimizer_failures": failures,
        }
    except Exception as exc:  # preserved as evidence; batch continues
        failure_path = output_path.with_suffix(".failure.json")
        write_json(
            failure_path,
            {
                "condition_id": condition_id,
                "method": method,
                "seed": seed,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - start,
            },
        )
        return {"status": "failed", "path": str(failure_path), "error": str(exc)}


def _diagnostic_filename(window_index: int, method: str, seed: int, kappa: float) -> str:
    kappa_token = str(kappa).replace(".", "p")
    return f"window{window_index:03d}__{method}__seed{seed}__kappa{kappa_token}.json"


def _run_diagnostic_task(
    data_path: Path,
    window_index: int,
    method: str,
    seed: int,
    portfolio: dict[str, Any],
    optimizer_config: dict[str, Any],
    train_days: int,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        return {"status": "skipped_existing", "path": str(output_path)}
    start = time.perf_counter()
    try:
        frame = load_processed_dataset(data_path)
        windows = build_monthly_schedule(frame.index, train_days)
        window = windows[window_index]
        train = frame.iloc[window.train_positions].to_numpy()
        pretrade = np.zeros(frame.shape[1])
        objective = OmegaObjective(
            train,
            pretrade=pretrade,
            cash_pre=1.0,
            cost_rate=float(portfolio["transaction_cost_bps"]) / 10_000.0,
            amortization_days=int(portfolio["cost_amortization_days"]),
            threshold=float(portfolio["omega_threshold"]),
            epsilon=float(portfolio["epsilon"]),
        )
        init_seed = int(
            np.random.SeedSequence([seed, window_index, 1709]).generate_state(1)[0]
        )
        method_seed = int(
            np.random.SeedSequence([seed, window_index, 2909]).generate_state(1)[0]
        )
        initial = random_feasible_population(
            np.random.default_rng(init_seed),
            int(optimizer_config["population_size"]),
            frame.shape[1],
            int(portfolio["cardinality"]),
            float(portfolio["lower_bound"]),
            float(portfolio["upper_bound"]),
        )
        result = run_method(
            method,
            objective,
            portfolio,
            optimizer_config,
            method_seed,
            initial,
        )
        payload = {
            "window_index": window_index,
            "train_start": window.train_start,
            "train_end": window.train_end,
            "hold_start_not_evaluated": window.hold_start,
            "method": method,
            "seed": seed,
            "init_seed": init_seed,
            "method_seed": method_seed,
            "kappa": float(optimizer_config["kappa"]),
            "formation_task": True,
            "optimizer": result.to_record(),
            "task_runtime_seconds": time.perf_counter() - start,
            "data_sha256": sha256_file(data_path),
        }
        write_json(output_path, payload)
        return {
            "status": "completed",
            "path": str(output_path),
            "runtime_seconds": payload["task_runtime_seconds"],
            "optimizer_failures": int(not result.success),
        }
    except Exception as exc:  # preserved as evidence; batch continues
        failure_path = output_path.with_suffix(".failure.json")
        write_json(
            failure_path,
            {
                "window_index": window_index,
                "method": method,
                "seed": seed,
                "kappa": float(optimizer_config["kappa"]),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "runtime_seconds": time.perf_counter() - start,
            },
        )
        return {"status": "failed", "path": str(failure_path), "error": str(exc)}


def _phase_summary(phase: str, records: list[dict[str, Any]], elapsed: float) -> dict[str, Any]:
    completed = [row for row in records if row["status"] == "completed"]
    skipped = [row for row in records if row["status"] == "skipped_existing"]
    failed = [row for row in records if row["status"] == "failed"]
    return {
        "phase": phase,
        "planned_tasks": len(records),
        "completed_tasks": len(completed),
        "skipped_existing": len(skipped),
        "failed_tasks": len(failed),
        "failure_records": [row["path"] for row in failed],
        "optimizer_failures": int(sum(row.get("optimizer_failures", 0) for row in records)),
        "phase_wall_seconds": elapsed,
        "sum_task_seconds": float(sum(row.get("runtime_seconds", 0.0) for row in completed)),
        "environment": environment_record(),
    }


def run_batch_phase(
    config: dict[str, Any],
    project_root: Path,
    phase: str,
    batch_id: str,
    config_path: Path | None = None,
) -> tuple[Path, dict[str, Any]]:
    if phase not in {"main", "diagnostics", "robustness"}:
        raise ValueError("phase must be main, diagnostics, or robustness")
    batch_root = project_root / "results" / "batch" / batch_id
    phase_root = batch_root / phase
    phase_root.mkdir(parents=True, exist_ok=True)
    n_jobs = int(config["execution"]["n_jobs"])
    train_days = int(config["walk_forward"]["train_days"])
    start = time.perf_counter()
    jobs = []

    if phase == "main":
        condition = config["main"]
        condition_id = condition["condition_id"]
        universe = int(condition["universe"])
        data_path = _dataset_path(project_root, config, universe)
        portfolio = _condition_portfolio(config, condition)
        for method in config["execution"]["stochastic_methods"]:
            for seed in config["execution"]["main_seeds"]:
                output = phase_root / _path_filename(condition_id, method, int(seed))
                jobs.append(
                    delayed(_run_path_task)(
                        data_path,
                        condition_id,
                        method,
                        int(seed),
                        portfolio,
                        config["optimizers"],
                        train_days,
                        output,
                    )
                )
        for method in config["execution"]["deterministic_methods"]:
            output = phase_root / _path_filename(condition_id, method, 0)
            jobs.append(
                delayed(_run_path_task)(
                    data_path,
                    condition_id,
                    method,
                    0,
                    portfolio,
                    config["optimizers"],
                    train_days,
                    output,
                )
            )

    elif phase == "robustness":
        for condition in config["robustness"]:
            condition_id = condition["condition_id"]
            universe = int(condition["universe"])
            data_path = _dataset_path(project_root, config, universe)
            portfolio = _condition_portfolio(config, condition)
            for method in config["robustness_methods"]["stochastic"]:
                for seed in config["execution"]["robustness_seeds"]:
                    output = phase_root / _path_filename(condition_id, method, int(seed))
                    jobs.append(
                        delayed(_run_path_task)(
                            data_path,
                            condition_id,
                            method,
                            int(seed),
                            portfolio,
                            config["optimizers"],
                            train_days,
                            output,
                        )
                    )
            for method in config["robustness_methods"]["deterministic"]:
                output = phase_root / _path_filename(condition_id, method, 0)
                jobs.append(
                    delayed(_run_path_task)(
                        data_path,
                        condition_id,
                        method,
                        0,
                        portfolio,
                        config["optimizers"],
                        train_days,
                        output,
                    )
                )

    else:
        condition = config["main"]
        universe = int(condition["universe"])
        data_path = _dataset_path(project_root, config, universe)
        frame = load_processed_dataset(data_path)
        windows = build_monthly_schedule(frame.index, train_days)
        selected = evenly_spaced_window_indices(
            len(windows), int(config["execution"]["diagnostic_dates"])
        )
        portfolio = _condition_portfolio(config, condition)
        seeds = range(
            int(config["execution"]["diagnostic_seeds"]["start"]),
            int(config["execution"]["diagnostic_seeds"]["end"]) + 1,
        )
        for window_index in selected:
            for method in config["execution"]["stochastic_methods"]:
                for seed in seeds:
                    output = phase_root / _diagnostic_filename(
                        int(window_index), method, int(seed), float(config["optimizers"]["kappa"])
                    )
                    jobs.append(
                        delayed(_run_diagnostic_task)(
                            data_path,
                            int(window_index),
                            method,
                            int(seed),
                            portfolio,
                            config["optimizers"],
                            train_days,
                            output,
                        )
                    )
        sensitivity_seeds = range(
            int(config["execution"]["kappa_sensitivity_seeds"]["start"]),
            int(config["execution"]["kappa_sensitivity_seeds"]["end"]) + 1,
        )
        for kappa in (2.0, 8.0):
            optimizer_config = deepcopy(config["optimizers"])
            optimizer_config["kappa"] = kappa
            for window_index in selected:
                for seed in sensitivity_seeds:
                    output = phase_root / _diagnostic_filename(
                        int(window_index), "rsf_abc", int(seed), kappa
                    )
                    jobs.append(
                        delayed(_run_diagnostic_task)(
                            data_path,
                            int(window_index),
                            "rsf_abc",
                            int(seed),
                            portfolio,
                            optimizer_config,
                            train_days,
                            output,
                        )
                    )

    records = Parallel(n_jobs=n_jobs, backend="loky", verbose=10)(jobs)
    summary = _phase_summary(phase, records, time.perf_counter() - start)
    summary["batch_id"] = batch_id
    resolved_config = (
        Path(config_path).resolve()
        if config_path is not None
        else (project_root / "configs" / "confirmatory.yaml").resolve()
    )
    summary["config_path"] = str(resolved_config)
    summary["config_sha256"] = sha256_file(resolved_config)
    summary_path = batch_root / f"{phase}_summary.json"
    write_json(summary_path, summary)
    return summary_path, summary
