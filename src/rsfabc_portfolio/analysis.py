"""Confirmatory result ingestion, inference, and publication figures."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .data import sha256_file
from .metrics import performance_metrics
from .provenance import environment_record, write_json


METHOD_ORDER = [
    "rsf_abc",
    "standard_abc",
    "ht_abc",
    "rs_light_abc",
    "pso",
    "de",
    "ew_n",
    "ew_k",
    "mv_k",
    "omega_milp",
]
ROBUSTNESS_METHOD_ORDER = [
    "rsf_abc",
    "standard_abc",
    "de",
    "ew_n",
    "ew_k",
    "mv_k",
    "omega_milp",
]
METHOD_LABELS = {
    "rsf_abc": "RSF-ABC",
    "standard_abc": "Standard ABC",
    "ht_abc": "HT-ABC",
    "rs_light_abc": "RS-Light-ABC",
    "pso": "PSO",
    "de": "DE",
    "ew_n": "EW-N",
    "ew_k": "EW-K",
    "mv_k": "MV-K",
    "omega_milp": "Omega-MILP",
}
METHOD_COLORS = {
    "rsf_abc": "#1F5A94",
    "standard_abc": "#D97706",
    "ht_abc": "#B8860B",
    "rs_light_abc": "#B24C7A",
    "pso": "#6B7D33",
    "de": "#6B7280",
    "ew_n": "#111827",
    "ew_k": "#4B5563",
    "mv_k": "#9CA3AF",
    "omega_milp": "#374151",
}
METHOD_LINESTYLES = {
    "rsf_abc": "-",
    "standard_abc": "--",
    "ht_abc": ":",
    "rs_light_abc": "-.",
    "pso": (0, (5, 2)),
    "de": (0, (2, 2)),
    "ew_n": "-",
    "ew_k": "--",
    "mv_k": ":",
    "omega_milp": "-.",
}
CONDITION_LABELS = {
    "scale_n10_k5": "N=10, K=5",
    "scale_n30_k10": "N=30, K=10",
    "cardinality_n49_k5": "N=49, K=5",
    "cardinality_n49_k15": "N=49, K=15",
    "cost_n49_c10": "N=49, cost=10 bps",
    "cost_n49_c50": "N=49, cost=50 bps",
    "threshold_n49_equal_weight_mean": "N=49, EW-mean threshold",
}
CONDITION_ORDER = list(CONDITION_LABELS)

PATH_METRICS = [
    "observations",
    "cumulative_return",
    "annualized_return",
    "annualized_volatility",
    "sharpe_zero_rf",
    "sortino_zero_threshold",
    "max_drawdown",
    "cvar95_daily_loss",
    "omega_zero_threshold",
    "terminal_wealth",
    "mean_monthly_turnover",
    "annualized_turnover",
    "total_one_way_turnover",
    "mean_rebalance_cost_rate",
    "total_rebalance_cost_rate",
    "mean_optimizer_runtime_seconds",
    "total_optimizer_runtime_seconds",
    "mean_training_objective",
    "optimizer_failure_rate",
    "signed_fallback_windows",
    "time_limited_feasible_windows",
]
BOOTSTRAP_METRICS = [
    "omega_zero_threshold",
    "annualized_return",
    "annualized_volatility",
    "sharpe_zero_rf",
    "sortino_zero_threshold",
    "max_drawdown",
    "cvar95_daily_loss",
    "cumulative_return",
    "terminal_wealth",
]
STOCHASTIC_METHODS = {
    "rsf_abc",
    "standard_abc",
    "ht_abc",
    "rs_light_abc",
    "pso",
    "de",
}


def _active_json_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(directory).glob("*.json")
        if not path.name.endswith(".failure.json")
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _relative(path: Path, project_root: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(project_root).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def path_metric_record(
    payload: dict[str, Any], source_path: Path, phase: str
) -> dict[str, Any]:
    metrics = performance_metrics(np.asarray(payload["net_returns"], dtype=float))
    rebalances = payload["rebalances"]
    turnovers = np.asarray([row["turnover"] for row in rebalances], dtype=float)
    costs = np.asarray([row["transaction_cost"] for row in rebalances], dtype=float)
    runtimes = np.asarray(
        [row["optimizer"]["runtime_seconds"] for row in rebalances], dtype=float
    )
    objectives = np.asarray(
        [row["optimizer"]["objective"] for row in rebalances], dtype=float
    )
    failures = [row for row in rebalances if not row["optimizer"]["success"]]
    signed = [
        row
        for row in rebalances
        if row["optimizer"].get("diagnostics", {}).get("signed_fallback_used", False)
    ]
    record: dict[str, Any] = {
        "phase": phase,
        "run_id": source_path.stem,
        "condition_id": payload["condition_id"],
        "universe": int(payload["universe"]),
        "method": payload["method"],
        "method_label": METHOD_LABELS[payload["method"]],
        "seed": int(payload["seed"]),
        "status": "partial" if failures else "success",
        "task_runtime_seconds": float(payload["task_runtime_seconds"]),
        "data_sha256": payload["data_sha256"],
        "oos_start": payload["daily_dates"][0],
        "oos_end": payload["daily_dates"][-1],
        "source_path": str(source_path),
    }
    record.update(metrics)
    record.update(
        {
            "mean_monthly_turnover": float(turnovers.mean()),
            "annualized_turnover": float(12.0 * turnovers.mean()),
            "total_one_way_turnover": float(turnovers.sum()),
            "mean_rebalance_cost_rate": float(costs.mean()),
            "total_rebalance_cost_rate": float(costs.sum()),
            "mean_optimizer_runtime_seconds": float(runtimes.mean()),
            "total_optimizer_runtime_seconds": float(runtimes.sum()),
            "mean_training_objective": float(objectives.mean()),
            "optimizer_failure_rate": float(len(failures) / len(rebalances)),
            "signed_fallback_windows": float(len(signed)),
            "time_limited_feasible_windows": float(len(failures)),
        }
    )
    return record


def ingest_path_phase(directory: Path, phase: str) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    payloads: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for path in _active_json_paths(directory):
        payload = _load_json(path)
        payloads.append(payload)
        records.append(path_metric_record(payload, path, phase))
    frame = pd.DataFrame.from_records(records)
    if frame.empty:
        raise ValueError(f"No active path results found in {directory}")
    return frame, payloads


def ingest_diagnostics(
    directory: Path,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    payloads: list[dict[str, Any]] = []
    for path in _active_json_paths(directory):
        payload = _load_json(path)
        payloads.append(payload)
        optimizer = payload["optimizer"]
        rows.append(
            {
                "run_id": path.stem,
                "window_index": int(payload["window_index"]),
                "train_start": payload["train_start"],
                "train_end": payload["train_end"],
                "hold_start_not_evaluated": payload["hold_start_not_evaluated"],
                "method": payload["method"],
                "method_label": METHOD_LABELS[payload["method"]],
                "seed": int(payload["seed"]),
                "init_seed": int(payload["init_seed"]),
                "method_seed": int(payload["method_seed"]),
                "kappa": float(payload["kappa"]),
                "training_omega_objective": float(optimizer["objective"]),
                "optimizer_runtime_seconds": float(optimizer["runtime_seconds"]),
                "task_runtime_seconds": float(payload["task_runtime_seconds"]),
                "evaluations": int(optimizer["evaluations"]),
                "success": bool(optimizer["success"]),
                "data_sha256": payload["data_sha256"],
                "source_path": str(path),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if frame.empty:
        raise ValueError(f"No diagnostic results found in {directory}")
    return frame, payloads


def method_metric_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (condition, method), group in frame.groupby(
        ["condition_id", "method"], sort=False
    ):
        for metric in PATH_METRICS:
            values = group[metric].to_numpy(dtype=float)
            rows.append(
                {
                    "condition_id": condition,
                    "method": method,
                    "method_label": METHOD_LABELS[method],
                    "metric": metric,
                    "n_paths": int(values.size),
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "std": float(values.std(ddof=1)) if values.size > 1 else np.nan,
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                }
            )
    return pd.DataFrame.from_records(rows)


def exact_paired_wilcoxon(differences: Iterable[float]) -> tuple[float, float, float]:
    """Exact two-sided signed-rank test and matched rank-biserial effect."""
    values = np.asarray(list(differences), dtype=float)
    values = values[np.isfinite(values)]
    values = values[np.abs(values) > 1e-15]
    if values.size == 0:
        return 0.0, 1.0, 0.0
    ranks = rankdata(np.abs(values), method="average")
    w_plus = float(ranks[values > 0].sum())
    total = float(ranks.sum())
    possible = np.empty(1 << values.size, dtype=float)
    for mask in range(1 << values.size):
        selected = [(mask >> index) & 1 for index in range(values.size)]
        possible[mask] = float(ranks[np.asarray(selected, dtype=bool)].sum())
    tolerance = 1e-12
    lower = float(np.mean(possible <= w_plus + tolerance))
    upper = float(np.mean(possible >= w_plus - tolerance))
    p_value = min(1.0, 2.0 * min(lower, upper))
    effect = float((2.0 * w_plus - total) / total)
    return w_plus, p_value, effect


def holm_adjust(p_values: Iterable[float]) -> np.ndarray:
    values = np.asarray(list(p_values), dtype=float)
    if values.ndim != 1 or values.size == 0:
        return values.copy()
    order = np.argsort(values)
    adjusted_sorted = np.empty(values.size, dtype=float)
    running = 0.0
    count = values.size
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[original_index])
        running = max(running, candidate)
        adjusted_sorted[rank] = running
    adjusted = np.empty(values.size, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def moving_block_indices(
    rng: np.random.Generator, observations: int, block_length: int
) -> np.ndarray:
    if observations <= 0 or block_length <= 0 or block_length > observations:
        raise ValueError("invalid moving-block dimensions")
    blocks = int(math.ceil(observations / block_length))
    starts = rng.integers(0, observations - block_length + 1, size=blocks)
    offsets = np.arange(block_length)
    return (starts[:, None] + offsets[None, :]).reshape(-1)[:observations]


def _return_map(payloads: list[dict[str, Any]]) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for payload in payloads:
        result[payload["method"]][int(payload["seed"])] = np.asarray(
            payload["net_returns"], dtype=float
        )
    return dict(result)


def paired_block_bootstrap(
    main_payloads: list[dict[str, Any]],
    primary_method: str,
    comparators: list[str],
    replicates: int,
    block_length: int,
    random_seed: int,
) -> pd.DataFrame:
    returns = _return_map(main_payloads)
    primary_seeds = sorted(returns[primary_method])
    if not primary_seeds:
        raise ValueError("primary method has no paths")
    rows: list[dict[str, Any]] = []
    primary_observed = {
        metric: float(
            np.mean(
                [
                    performance_metrics(returns[primary_method][seed])[metric]
                    for seed in primary_seeds
                ]
            )
        )
        for metric in BOOTSTRAP_METRICS
    }

    for comparator_index, comparator in enumerate(comparators):
        comparator_seeds = sorted(returns[comparator])
        if comparator in STOCHASTIC_METHODS:
            paired_seeds = sorted(set(primary_seeds) & set(comparator_seeds))
            if not paired_seeds:
                raise ValueError(f"No common seeds for {primary_method} and {comparator}")
        else:
            paired_seeds = primary_seeds
            if comparator_seeds != [0]:
                raise ValueError(f"Deterministic comparator seed mismatch: {comparator}")
        comparator_observed = {
            metric: float(
                np.mean(
                    [
                        performance_metrics(returns[comparator][seed])[metric]
                        for seed in comparator_seeds
                    ]
                )
            )
            for metric in BOOTSTRAP_METRICS
        }
        distributions = {
            metric: np.empty(replicates, dtype=float) for metric in BOOTSTRAP_METRICS
        }
        rng = np.random.default_rng(
            np.random.SeedSequence([random_seed, comparator_index, 881])
        )
        observations = len(returns[primary_method][primary_seeds[0]])
        for replicate in range(replicates):
            seed = int(rng.choice(paired_seeds))
            primary_path = returns[primary_method][seed]
            comparator_seed = seed if comparator in STOCHASTIC_METHODS else 0
            comparator_path = returns[comparator][comparator_seed]
            indices = moving_block_indices(rng, observations, block_length)
            primary_metrics = performance_metrics(primary_path[indices])
            comparator_metrics = performance_metrics(comparator_path[indices])
            for metric in BOOTSTRAP_METRICS:
                distributions[metric][replicate] = (
                    primary_metrics[metric] - comparator_metrics[metric]
                )
        for metric in BOOTSTRAP_METRICS:
            distribution = distributions[metric]
            lower, upper = np.quantile(distribution, [0.025, 0.975])
            lower_tail = (np.count_nonzero(distribution <= 0.0) + 1.0) / (
                replicates + 1.0
            )
            upper_tail = (np.count_nonzero(distribution >= 0.0) + 1.0) / (
                replicates + 1.0
            )
            rows.append(
                {
                    "primary_method": primary_method,
                    "comparator": comparator,
                    "metric": metric,
                    "estimate_primary_minus_comparator": float(
                        primary_observed[metric] - comparator_observed[metric]
                    ),
                    "bootstrap_mean_difference": float(distribution.mean()),
                    "ci_lower_95": float(lower),
                    "ci_upper_95": float(upper),
                    "two_sided_sign_p": float(min(1.0, 2.0 * min(lower_tail, upper_tail))),
                    "holm_adjusted_p_primary_family": np.nan,
                    "bootstrap_replicates": int(replicates),
                    "block_length": int(block_length),
                    "seed_sampling": (
                        "matched stochastic seed"
                        if comparator in STOCHASTIC_METHODS
                        else "sampled RSF seed; deterministic comparator fixed"
                    ),
                }
            )
    frame = pd.DataFrame.from_records(rows)
    primary_mask = frame["metric"] == "omega_zero_threshold"
    frame.loc[primary_mask, "holm_adjusted_p_primary_family"] = holm_adjust(
        frame.loc[primary_mask, "two_sided_sign_p"].to_numpy(dtype=float)
    )
    return frame


def diagnostic_tables(
    diagnostics: pd.DataFrame,
    diagnostic_payloads: list[dict[str, Any]],
    comparators: list[str],
    grid_start: int,
    grid_step: int,
    grid_end: int,
) -> dict[str, pd.DataFrame]:
    main = diagnostics[np.isclose(diagnostics["kappa"], 4.0)].copy()
    date_summary = (
        main.groupby(["window_index", "train_end", "method"], as_index=False)
        .agg(
            median_training_omega=("training_omega_objective", "median"),
            q25_training_omega=("training_omega_objective", lambda x: x.quantile(0.25)),
            q75_training_omega=("training_omega_objective", lambda x: x.quantile(0.75)),
            mean_training_omega=("training_omega_objective", "mean"),
            std_training_omega=("training_omega_objective", "std"),
            median_runtime_seconds=("optimizer_runtime_seconds", "median"),
            n_seeds=("seed", "nunique"),
        )
    )
    pivot = date_summary.pivot(
        index="window_index", columns="method", values="median_training_omega"
    )
    test_rows: list[dict[str, Any]] = []
    for comparator in comparators:
        differences = pivot["rsf_abc"] - pivot[comparator]
        statistic, p_value, effect = exact_paired_wilcoxon(differences)
        test_rows.append(
            {
                "primary_method": "rsf_abc",
                "comparator": comparator,
                "n_dates": int(differences.size),
                "median_date_difference": float(np.median(differences)),
                "wilcoxon_w_plus": statistic,
                "exact_two_sided_p": p_value,
                "holm_adjusted_p": np.nan,
                "matched_rank_biserial": effect,
            }
        )
    pairwise = pd.DataFrame.from_records(test_rows)
    pairwise["holm_adjusted_p"] = holm_adjust(pairwise["exact_two_sided_p"])

    fair_kappa = diagnostics[
        (diagnostics["method"] == "rsf_abc") & (diagnostics["seed"] <= 1010)
    ].copy()
    kappa_dates = (
        fair_kappa.groupby(["window_index", "train_end", "kappa"], as_index=False)
        .agg(
            median_training_omega=("training_omega_objective", "median"),
            q25_training_omega=("training_omega_objective", lambda x: x.quantile(0.25)),
            q75_training_omega=("training_omega_objective", lambda x: x.quantile(0.75)),
            n_seeds=("seed", "nunique"),
        )
    )
    kappa_pivot = kappa_dates.pivot(
        index="window_index", columns="kappa", values="median_training_omega"
    )
    kappa_tests: list[dict[str, Any]] = []
    for comparator_kappa in (2.0, 8.0):
        differences = kappa_pivot[4.0] - kappa_pivot[comparator_kappa]
        statistic, p_value, effect = exact_paired_wilcoxon(differences)
        kappa_tests.append(
            {
                "primary_kappa": 4.0,
                "comparator_kappa": comparator_kappa,
                "n_dates": int(differences.size),
                "median_date_difference": float(np.median(differences)),
                "wilcoxon_w_plus": statistic,
                "exact_two_sided_p": p_value,
                "holm_adjusted_p": np.nan,
                "matched_rank_biserial": effect,
            }
        )
    kappa_pairwise = pd.DataFrame.from_records(kappa_tests)
    kappa_pairwise["holm_adjusted_p"] = holm_adjust(
        kappa_pairwise["exact_two_sided_p"]
    )

    grid = np.unique(
        np.concatenate(
            (
                np.array([grid_start]),
                np.arange(grid_step, grid_end + grid_step, grid_step),
            )
        )
    )
    convergence_values: dict[tuple[str, int], list[float]] = defaultdict(list)
    for payload in diagnostic_payloads:
        if not np.isclose(float(payload["kappa"]), 4.0):
            continue
        optimizer = payload["optimizer"]
        evaluations = np.asarray(optimizer["convergence_evaluations"], dtype=int)
        best = np.asarray(optimizer["convergence_best"], dtype=float)
        indices = np.searchsorted(evaluations, grid, side="right") - 1
        for evaluation, index in zip(grid, indices, strict=True):
            if index >= 0:
                convergence_values[(payload["method"], int(evaluation))].append(
                    float(best[index])
                )
    convergence_rows: list[dict[str, Any]] = []
    for (method, evaluation), values in convergence_values.items():
        array = np.asarray(values, dtype=float)
        convergence_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "evaluations": evaluation,
                "median_best_training_omega": float(np.median(array)),
                "q25_best_training_omega": float(np.quantile(array, 0.25)),
                "q75_best_training_omega": float(np.quantile(array, 0.75)),
                "n_runs": int(array.size),
            }
        )
    convergence = pd.DataFrame.from_records(convergence_rows).sort_values(
        ["method", "evaluations"]
    )
    runtime = (
        main.groupby("method", as_index=False)
        .agg(
            n_runs=("run_id", "count"),
            median_runtime_seconds=("optimizer_runtime_seconds", "median"),
            mean_runtime_seconds=("optimizer_runtime_seconds", "mean"),
            q25_runtime_seconds=("optimizer_runtime_seconds", lambda x: x.quantile(0.25)),
            q75_runtime_seconds=("optimizer_runtime_seconds", lambda x: x.quantile(0.75)),
            failure_rate=("success", lambda x: 1.0 - float(np.mean(x))),
        )
    )
    return {
        "diagnostic_date_summary": date_summary,
        "diagnostic_pairwise_tests": pairwise,
        "kappa_date_summary": kappa_dates,
        "kappa_pairwise_tests": kappa_pairwise,
        "diagnostic_convergence_summary": convergence,
        "diagnostic_runtime_summary": runtime,
    }


def experiment_matrix(
    main_metrics: pd.DataFrame,
    robustness_metrics: pd.DataFrame,
    diagnostics: pd.DataFrame,
    project_root: Path,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    machine = "none; macOS-26.2-arm64; 8 logical CPUs; 6 workers"
    for _, record in pd.concat(
        [main_metrics, robustness_metrics], ignore_index=True
    ).iterrows():
        notes = (
            f"{int(record['time_limited_feasible_windows'])}/215 optimizer windows "
            "used time-limited feasible incumbents"
            if record["status"] == "partial"
            else "complete validated walk-forward path"
        )
        for metric in PATH_METRICS:
            rows.append(
                {
                    "run_id": record["run_id"],
                    "paper_id": "rsfabc_portfolio",
                    "commit_hash": "unavailable-not-a-git-repository",
                    "dataset": f"French_value_weighted_{int(record['universe'])}_industry_daily",
                    "split": f"walk_forward_oos_{record['oos_start']}_to_{record['oos_end']}",
                    "model": record["method"],
                    "config": f"configs/confirmatory.yaml#{record['condition_id']}",
                    "seed": int(record["seed"]),
                    "metric": metric,
                    "value": float(record[metric]),
                    "paper_result": "",
                    "delta": "",
                    "status": record["status"],
                    "error_type": "unknown" if record["status"] == "partial" else "",
                    "runtime": float(record["task_runtime_seconds"]),
                    "gpu": machine,
                    "notes": notes,
                    "log_path": _relative(Path(record["source_path"]), project_root),
                }
            )
    diagnostic_metrics = [
        "training_omega_objective",
        "optimizer_runtime_seconds",
        "task_runtime_seconds",
        "evaluations",
        "success",
    ]
    for _, record in diagnostics.iterrows():
        for metric in diagnostic_metrics:
            value = float(record[metric])
            rows.append(
                {
                    "run_id": record["run_id"],
                    "paper_id": "rsfabc_portfolio",
                    "commit_hash": "unavailable-not-a-git-repository",
                    "dataset": "French_value_weighted_49_industry_daily",
                    "split": (
                        f"formation_only_train_{record['train_start']}_to_"
                        f"{record['train_end']}; hold_not_evaluated"
                    ),
                    "model": record["method"],
                    "config": f"configs/confirmatory.yaml#diagnostics_kappa_{record['kappa']:g}",
                    "seed": int(record["seed"]),
                    "metric": metric,
                    "value": value,
                    "paper_result": "",
                    "delta": "",
                    "status": "success" if record["success"] else "failed",
                    "error_type": "" if record["success"] else "unknown",
                    "runtime": float(record["task_runtime_seconds"]),
                    "gpu": machine,
                    "notes": (
                        f"formation-only diagnostic; window={int(record['window_index'])}; "
                        "no holding-period performance inspected"
                    ),
                    "log_path": _relative(Path(record["source_path"]), project_root),
                }
            )
    columns = [
        "run_id",
        "paper_id",
        "commit_hash",
        "dataset",
        "split",
        "model",
        "config",
        "seed",
        "metric",
        "value",
        "paper_result",
        "delta",
        "status",
        "error_type",
        "runtime",
        "gpu",
        "notes",
        "log_path",
    ]
    return pd.DataFrame.from_records(rows, columns=columns)


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 12,
            "axes.labelsize": 9,
            "axes.edgecolor": "#374151",
            "axes.labelcolor": "#1F2937",
            "xtick.color": "#374151",
            "ytick.color": "#374151",
            "text.color": "#111827",
            "grid.color": "#E5E7EB",
            "grid.linewidth": 0.7,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _title_with_subtitle(ax: plt.Axes, title: str, subtitle: str) -> None:
    ax.set_title(title, loc="left", pad=34, fontweight="bold")
    ax.text(
        0,
        1.012,
        subtitle,
        transform=ax.transAxes,
        fontsize=8.5,
        color="#4B5563",
        ha="left",
        va="bottom",
    )


def _save_figure(
    fig: plt.Figure, base_path: Path, formats: list[str], dpi: int
) -> list[str]:
    outputs: list[str] = []
    for extension in formats:
        path = base_path.with_suffix(f".{extension}")
        fig.savefig(
            path,
            dpi=dpi if extension.lower() == "png" else None,
            bbox_inches="tight",
            facecolor="white",
        )
        outputs.append(str(path))
    plt.close(fig)
    return outputs


def create_figures(
    main_payloads: list[dict[str, Any]],
    main_metrics: pd.DataFrame,
    robustness_metrics: pd.DataFrame,
    diagnostic_date_summary: pd.DataFrame,
    convergence: pd.DataFrame,
    figure_dir: Path,
    formats: list[str],
    dpi: int,
) -> list[str]:
    _style()
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []
    returns = _return_map(main_payloads)
    dates = pd.to_datetime(main_payloads[0]["daily_dates"])

    fig, ax = plt.subplots(figsize=(11.2, 6.3))
    for method in METHOD_ORDER:
        wealth = np.vstack(
            [np.cumprod(1.0 + values) for values in returns[method].values()]
        )
        median = np.median(wealth, axis=0)
        q25 = np.quantile(wealth, 0.25, axis=0)
        q75 = np.quantile(wealth, 0.75, axis=0)
        ax.plot(
            dates,
            median,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.2 if method == "rsf_abc" else 1.35,
            alpha=1.0 if method == "rsf_abc" else 0.88,
        )
        if wealth.shape[0] > 1:
            ax.fill_between(
                dates,
                q25,
                q75,
                color=METHOD_COLORS[method],
                alpha=0.055,
                linewidth=0,
            )
    ax.set_yscale("log")
    _title_with_subtitle(
        ax,
        "Net cumulative wealth across the main comparison",
        "49 industries; 25 bps; February 2008–December 2025; median and IQR across seeds; log wealth axis",
    )
    ax.set_ylabel("Wealth (initial value = 1; log scale)")
    ax.set_xlabel("Out-of-sample date")
    ax.xaxis.set_major_locator(mdates.YearLocator(3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(True, which="major", axis="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, loc="upper left", fontsize=8)
    outputs.extend(
        _save_figure(fig, figure_dir / "fig1_main_cumulative_wealth", formats, dpi)
    )

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    max_value = float(main_metrics["omega_zero_threshold"].max())
    for position, method in enumerate(reversed(METHOD_ORDER)):
        values = main_metrics.loc[
            main_metrics["method"] == method, "omega_zero_threshold"
        ].to_numpy(dtype=float)
        ax.hlines(
            position,
            values.min(),
            values.max(),
            color="#9CA3AF",
            linewidth=1.1,
            zorder=1,
        )
        ax.scatter(
            values,
            np.full(values.size, position),
            s=30,
            facecolors="white",
            edgecolors=METHOD_COLORS[method],
            linewidths=1.1,
            zorder=2,
        )
        ax.scatter(
            [values.mean()],
            [position],
            marker="D",
            s=48,
            color=METHOD_COLORS[method],
            edgecolors="#111827",
            linewidths=0.4,
            zorder=3,
        )
        ax.text(
            values.mean() + max_value * 0.007,
            position,
            f"{values.mean():.3f}",
            va="center",
            ha="left",
            fontsize=7.8,
            color="#374151",
        )
    ax.set_yticks(range(len(METHOD_ORDER)))
    ax.set_yticklabels([METHOD_LABELS[m] for m in reversed(METHOD_ORDER)])
    ax.set_xlim(0.0, max_value * 1.08)
    ax.axvline(1.0, color="#6B7280", linestyle=":", linewidth=1.0)
    _title_with_subtitle(
        ax,
        "Net out-of-sample Omega by method and seed",
        "Diamonds show method means; open circles show seed paths; deterministic methods have one path; dotted line marks Omega = 1",
    )
    ax.set_xlabel("Omega ratio at zero daily threshold")
    ax.grid(True, axis="x")
    ax.spines[["top", "right"]].set_visible(False)
    outputs.extend(
        _save_figure(fig, figure_dir / "fig2_main_omega_comparison", formats, dpi)
    )

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    diagnostic_methods = METHOD_ORDER[:6]
    box_values = [
        diagnostic_date_summary.loc[
            diagnostic_date_summary["method"] == method,
            "median_training_omega",
        ].to_numpy(dtype=float)
        for method in diagnostic_methods
    ]
    boxes = ax.boxplot(
        box_values,
        patch_artist=True,
        widths=0.55,
        showfliers=False,
        medianprops={"color": "#111827", "linewidth": 1.3},
        whiskerprops={"color": "#6B7280"},
        capprops={"color": "#6B7280"},
    )
    for patch, method in zip(boxes["boxes"], diagnostic_methods, strict=True):
        patch.set_facecolor(METHOD_COLORS[method])
        patch.set_alpha(0.22)
        patch.set_edgecolor(METHOD_COLORS[method])
    for position, (method, values) in enumerate(
        zip(diagnostic_methods, box_values, strict=True), start=1
    ):
        offsets = np.linspace(-0.13, 0.13, values.size)
        ax.scatter(
            position + offsets,
            values,
            s=18,
            color=METHOD_COLORS[method],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.9,
        )
    ax.set_xticks(range(1, len(diagnostic_methods) + 1))
    ax.set_xticklabels([METHOD_LABELS[m] for m in diagnostic_methods], rotation=18)
    _title_with_subtitle(
        ax,
        "Fixed-budget training objective on diagnostic dates",
        "Twelve predeclared dates; each point is the median across 30 seeds; 3,000 evaluations per run",
    )
    ax.set_ylabel("Best feasible training Omega")
    ax.grid(True, axis="y")
    ax.spines[["top", "right"]].set_visible(False)
    outputs.extend(
        _save_figure(fig, figure_dir / "fig3_diagnostic_objective", formats, dpi)
    )

    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    for method in diagnostic_methods:
        group = convergence[convergence["method"] == method].sort_values("evaluations")
        x = group["evaluations"].to_numpy(dtype=float)
        median = group["median_best_training_omega"].to_numpy(dtype=float)
        q25 = group["q25_best_training_omega"].to_numpy(dtype=float)
        q75 = group["q75_best_training_omega"].to_numpy(dtype=float)
        ax.plot(
            x,
            median,
            label=METHOD_LABELS[method],
            color=METHOD_COLORS[method],
            linestyle=METHOD_LINESTYLES[method],
            linewidth=2.2 if method == "rsf_abc" else 1.5,
        )
        ax.fill_between(x, q25, q75, color=METHOD_COLORS[method], alpha=0.04)
    _title_with_subtitle(
        ax,
        "Median best-feasible objective by evaluation budget",
        "Twelve dates × 30 seeds per method; ribbons show the interquartile range across 360 runs",
    )
    ax.set_xlabel("Objective evaluations")
    ax.set_ylabel("Best feasible training Omega")
    ax.grid(True, axis="both")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(ncol=2, fontsize=8)
    outputs.extend(
        _save_figure(fig, figure_dir / "fig4_diagnostic_convergence", formats, dpi)
    )

    heat = (
        robustness_metrics.groupby(["condition_id", "method"])[
            "omega_zero_threshold"
        ]
        .mean()
        .unstack("method")
        .reindex(index=CONDITION_ORDER, columns=ROBUSTNESS_METHOD_ORDER)
    )
    values = heat.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    image = ax.imshow(values, cmap="Blues", aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            normalized = (values[row, column] - np.nanmin(values)) / max(
                np.nanmax(values) - np.nanmin(values), 1e-12
            )
            ax.text(
                column,
                row,
                f"{values[row, column]:.3f}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if normalized > 0.58 else "#111827",
            )
    ax.set_xticks(range(len(ROBUSTNESS_METHOD_ORDER)))
    ax.set_xticklabels(
        [METHOD_LABELS[m] for m in ROBUSTNESS_METHOD_ORDER], rotation=25, ha="right"
    )
    ax.set_yticks(range(len(CONDITION_ORDER)))
    ax.set_yticklabels([CONDITION_LABELS[c] for c in CONDITION_ORDER])
    _title_with_subtitle(
        ax,
        "Mean net out-of-sample Omega across robustness conditions",
        "Stochastic cells average three seed paths; deterministic cells use one path; all frozen conditions shown",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02)
    colorbar.set_label("Omega ratio")
    ax.tick_params(length=0)
    outputs.extend(
        _save_figure(fig, figure_dir / "fig5_robustness_omega_heatmap", formats, dpi)
    )
    return outputs


def run_confirmatory_analysis(
    project_root: Path, analysis_config: dict[str, Any], analysis_config_path: Path
) -> tuple[Path, dict[str, Any]]:
    started = time.perf_counter()
    project_root = Path(project_root)
    settings = analysis_config["analysis"]
    batch_id = settings["batch_id"]
    batch_root = project_root / "results" / "batch" / batch_id
    output_root = project_root / settings["output_dir"]
    table_dir = output_root / "tables"
    figure_dir = output_root / "figures"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    main_metrics, main_payloads = ingest_path_phase(batch_root / "main", "main")
    robustness_metrics, robustness_payloads = ingest_path_phase(
        batch_root / "robustness", "robustness"
    )
    diagnostics, diagnostic_payloads = ingest_diagnostics(batch_root / "diagnostics")

    main_metrics.to_csv(table_dir / "main_path_metrics.csv", index=False)
    robustness_metrics.to_csv(table_dir / "robustness_path_metrics.csv", index=False)
    diagnostics.to_csv(table_dir / "diagnostic_runs.csv", index=False)
    main_summary = method_metric_summary(main_metrics)
    robustness_summary = method_metric_summary(robustness_metrics)
    main_summary.to_csv(table_dir / "main_method_summary.csv", index=False)
    robustness_summary.to_csv(table_dir / "robustness_method_summary.csv", index=False)

    bootstrap = paired_block_bootstrap(
        main_payloads,
        settings["primary_method"],
        list(settings["primary_comparators"]),
        int(settings["bootstrap_replicates"]),
        int(settings["moving_block_length"]),
        int(settings["random_seed"]),
    )
    bootstrap.to_csv(table_dir / "main_paired_block_bootstrap.csv", index=False)

    diagnostic_outputs = diagnostic_tables(
        diagnostics,
        diagnostic_payloads,
        list(settings["diagnostic_comparators"]),
        int(settings["convergence_grid_start"]),
        int(settings["convergence_grid_step"]),
        int(settings["convergence_grid_end"]),
    )
    for name, frame in diagnostic_outputs.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)

    matrix = experiment_matrix(
        main_metrics, robustness_metrics, diagnostics, project_root
    )
    matrix_path = project_root / "matrices" / "experiment_matrix.csv"
    matrix.to_csv(matrix_path, index=False)

    figure_settings = analysis_config["figures"]
    figures = create_figures(
        main_payloads,
        main_metrics,
        robustness_metrics,
        diagnostic_outputs["diagnostic_date_summary"],
        diagnostic_outputs["diagnostic_convergence_summary"],
        figure_dir,
        [str(value) for value in figure_settings["formats"]],
        int(figure_settings["dpi"]),
    )

    manifest = {
        "analysis_version": analysis_config["project"]["analysis_version"],
        "batch_id": batch_id,
        "analysis_config_path": _relative(analysis_config_path, project_root),
        "analysis_config_sha256": sha256_file(analysis_config_path),
        "confirmatory_config_sha256": sha256_file(
            project_root / "configs" / "confirmatory.yaml"
        ),
        "milp_source_sha256": sha256_file(
            project_root / "src" / "rsfabc_portfolio" / "milp.py"
        ),
        "analysis_source_sha256": sha256_file(
            project_root / "src" / "rsfabc_portfolio" / "analysis.py"
        ),
        "metrics_source_sha256": sha256_file(
            project_root / "src" / "rsfabc_portfolio" / "metrics.py"
        ),
        "raw_counts": {
            "main_paths": len(main_payloads),
            "robustness_paths": len(robustness_payloads),
            "diagnostic_runs": len(diagnostic_payloads),
        },
        "experiment_matrix_rows": int(len(matrix)),
        "bootstrap": {
            "replicates_per_comparison": int(settings["bootstrap_replicates"]),
            "block_length": int(settings["moving_block_length"]),
            "random_seed": int(settings["random_seed"]),
            "primary_family": list(settings["primary_comparators"]),
        },
        "protocol_amendments": ["phase3_analysis/protocol_amendment_001.md"],
        "analysis_corrections": ["phase3_analysis/analysis_corrections.md"],
        "known_partial_baseline": {
            "condition": "scale_n10_k5",
            "method": "omega_milp",
            "time_limited_feasible_windows": 5,
            "total_windows": 215,
        },
        "tables": sorted(
            _relative(path, project_root) for path in table_dir.glob("*.csv")
        ),
        "figures": sorted(_relative(Path(path), project_root) for path in figures),
        "experiment_matrix": _relative(matrix_path, project_root),
        "environment": environment_record(),
        "wall_seconds": time.perf_counter() - started,
    }
    manifest_path = output_root / "analysis_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest
