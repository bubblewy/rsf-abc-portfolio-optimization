"""Evidence-repair analyses for the RSF-ABC revision round."""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .analysis import exact_paired_wilcoxon, holm_adjust
from .data import sha256_file
from .metrics import performance_metrics
from .provenance import environment_record, write_json


METHOD_LABELS = {
    "rsf_abc": "RSF-ABC",
    "standard_abc": "Standard ABC",
    "fixed_mix_abc": "Fixed-Mix-ABC",
}


def _active_json_paths(directory: Path) -> list[Path]:
    return sorted(
        path
        for path in Path(directory).glob("*.json")
        if not path.name.endswith(".failure.json")
    )


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def calibrate_fixed_mix(
    project_root: Path, batch_id: str
) -> tuple[Path, dict[str, Any]]:
    """Match a state-independent mixture to RSF's realized diagnostic branch rate."""
    diagnostic_dir = project_root / "results" / "batch" / batch_id / "diagnostics"
    paths = _active_json_paths(diagnostic_dir)
    payloads = [
        _load_json(path)
        for path in paths
        if "__rsf_abc__" in path.name and "__kappa4p0" in path.name
    ]
    if not payloads:
        raise ValueError(f"No RSF diagnostic records found in {diagnostic_dir}")

    heavy = 0
    conservative = 0
    p_weighted_sum = 0.0
    p_count = 0
    selected_dates: set[str] = set()
    for payload in payloads:
        optimizer = payload["optimizer"]
        diagnostics = optimizer["diagnostics"]
        counts = diagnostics["mode_counts"]
        heavy += int(counts["heavy_explore"])
        conservative += int(counts["conservative"])
        p_summary = diagnostics["proposal_p_explore_summary"]
        if p_summary is None:
            raise ValueError("RSF diagnostic record lacks p_explore summary")
        p_weighted_sum += float(p_summary["mean"]) * int(p_summary["count"])
        p_count += int(p_summary["count"])
        selected_dates.add(str(payload["train_end"]))

    governed_proposals = heavy + conservative
    if governed_proposals <= 0 or p_count <= 0:
        raise ValueError("RSF diagnostic records contain no governed proposals")
    realized_rate = heavy / governed_proposals
    expected_rate = p_weighted_sum / p_count
    record = {
        "calibration_rule": (
            "fixed state-independent exploration probability equals the pooled "
            "realized RSF heavy-exploration frequency across the predeclared "
            "formation-only diagnostic runs"
        ),
        "uses_out_of_sample_performance": False,
        "batch_id": batch_id,
        "diagnostic_runs": len(payloads),
        "diagnostic_dates": sorted(selected_dates),
        "n_dates": len(selected_dates),
        "seeds_per_date": len({int(payload["seed"]) for payload in payloads}),
        "governed_proposals": governed_proposals,
        "heavy_explore_proposals": heavy,
        "conservative_proposals": conservative,
        "rsf_realized_heavy_explore_rate": realized_rate,
        "rsf_mean_model_probability": expected_rate,
        "fixed_explore_probability": realized_rate,
        "absolute_realized_minus_expected_rate": abs(realized_rate - expected_rate),
        "source_files": [path.name for path in paths if "__rsf_abc__" in path.name],
    }
    output = (
        project_root
        / "results"
        / "batch"
        / batch_id
        / "g5_fixed_mix_calibration.json"
    )
    write_json(output, record)
    return output, record


def _moving_block_indices(
    rng: np.random.Generator, observations: int, block_length: int
) -> np.ndarray:
    if observations <= 0 or block_length <= 0 or block_length > observations:
        raise ValueError("invalid moving-block dimensions")
    blocks = int(math.ceil(observations / block_length))
    starts = rng.integers(0, observations - block_length + 1, size=blocks)
    offsets = np.arange(block_length)
    return (starts[:, None] + offsets[None, :]).reshape(-1)[:observations]


def _net_omega(returns: np.ndarray) -> float:
    values = np.asarray(returns, dtype=float)
    gains = np.maximum(values, 0.0).mean()
    shortfalls = np.maximum(-values, 0.0).mean()
    return float(gains / (shortfalls + 1e-12))


def _return_map(
    payloads: list[dict[str, Any]],
) -> dict[str, dict[int, np.ndarray]]:
    result: dict[str, dict[int, np.ndarray]] = defaultdict(dict)
    for payload in payloads:
        result[str(payload["method"])][int(payload["seed"])] = np.asarray(
            payload["net_returns"], dtype=float
        )
    return dict(result)


def hierarchical_paired_bootstrap(
    main_payloads: list[dict[str, Any]],
    primary_method: str,
    comparators: list[str],
    replicates: int,
    block_length: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resample matched seeds, then paired time blocks within each sampled seed."""
    returns = _return_map(main_payloads)
    primary_seeds = sorted(returns.get(primary_method, {}))
    if not primary_seeds:
        raise ValueError("primary method has no paths")
    seed_rows: list[dict[str, Any]] = []
    bootstrap_rows: list[dict[str, Any]] = []

    for comparator_index, comparator in enumerate(comparators):
        paired_seeds = sorted(
            set(primary_seeds) & set(returns.get(comparator, {}))
        )
        if paired_seeds != primary_seeds:
            raise ValueError(f"Incomplete matched seeds for {comparator}: {paired_seeds}")
        lengths = {
            len(returns[method][seed])
            for method in (primary_method, comparator)
            for seed in paired_seeds
        }
        if len(lengths) != 1:
            raise ValueError(f"Unequal path lengths for {comparator}: {lengths}")
        observations = lengths.pop()

        effects = []
        for seed in paired_seeds:
            primary_omega = _net_omega(returns[primary_method][seed])
            comparator_omega = _net_omega(returns[comparator][seed])
            effect = primary_omega - comparator_omega
            effects.append(effect)
            seed_rows.append(
                {
                    "comparator": comparator,
                    "comparator_label": METHOD_LABELS.get(comparator, comparator),
                    "seed": seed,
                    "rsf_net_omega": primary_omega,
                    "comparator_net_omega": comparator_omega,
                    "rsf_minus_comparator": effect,
                    "observations": observations,
                }
            )

        rng = np.random.default_rng(
            np.random.SeedSequence([random_seed, comparator_index, 5519])
        )
        distribution = np.empty(replicates, dtype=float)
        for replicate in range(replicates):
            sampled_seeds = rng.choice(
                paired_seeds, size=len(paired_seeds), replace=True
            )
            replicate_effects = np.empty(len(paired_seeds), dtype=float)
            for index, sampled_seed in enumerate(sampled_seeds):
                sampled_seed = int(sampled_seed)
                block_indices = _moving_block_indices(
                    rng, observations, block_length
                )
                primary_omega = _net_omega(
                    returns[primary_method][sampled_seed][block_indices]
                )
                comparator_omega = _net_omega(
                    returns[comparator][sampled_seed][block_indices]
                )
                replicate_effects[index] = primary_omega - comparator_omega
            distribution[replicate] = float(replicate_effects.mean())

        lower, upper = np.quantile(distribution, [0.025, 0.975])
        lower_tail = (np.count_nonzero(distribution <= 0.0) + 1.0) / (
            replicates + 1.0
        )
        upper_tail = (np.count_nonzero(distribution >= 0.0) + 1.0) / (
            replicates + 1.0
        )
        bootstrap_rows.append(
            {
                "primary_method": primary_method,
                "comparator": comparator,
                "comparator_label": METHOD_LABELS.get(comparator, comparator),
                "estimand": "mean matched-seed complete-path net Omega difference",
                "estimate_primary_minus_comparator": float(np.mean(effects)),
                "bootstrap_mean_difference": float(distribution.mean()),
                "ci_lower_95": float(lower),
                "ci_upper_95": float(upper),
                "two_sided_sign_p": float(
                    min(1.0, 2.0 * min(lower_tail, upper_tail))
                ),
                "holm_adjusted_p": np.nan,
                "bootstrap_replicates": replicates,
                "block_length": block_length,
                "seed_clusters": len(paired_seeds),
                "daily_observations_per_seed": observations,
                "resampling_scheme": (
                    "paired seed clusters with replacement; paired moving blocks "
                    "within each sampled seed"
                ),
            }
        )

    bootstrap = pd.DataFrame.from_records(bootstrap_rows)
    bootstrap["holm_adjusted_p"] = holm_adjust(
        bootstrap["two_sided_sign_p"].to_numpy(dtype=float)
    )
    return pd.DataFrame.from_records(seed_rows), bootstrap


def _main_path_table(payloads: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        metrics = performance_metrics(np.asarray(payload["net_returns"], dtype=float))
        turnovers = np.asarray(
            [row["turnover"] for row in payload["rebalances"]], dtype=float
        )
        rows.append(
            {
                "method": payload["method"],
                "method_label": METHOD_LABELS.get(payload["method"], payload["method"]),
                "seed": int(payload["seed"]),
                **metrics,
                "mean_monthly_turnover": float(turnovers.mean()),
                "rebalances": len(payload["rebalances"]),
                "source_file": Path(payload["source_path"]).name,
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(["method", "seed"])


def _mechanism_tables(
    payloads: list[dict[str, Any]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for payload in payloads:
        diagnostics = payload["optimizer"]["diagnostics"]
        budget = diagnostics.get("proposal_budget_summary") or {}
        probability = diagnostics.get("proposal_p_explore_summary") or {}
        counts = diagnostics["mode_counts"]
        governed = int(counts["heavy_explore"]) + int(counts["conservative"])
        rows.append(
            {
                "method": payload["method"],
                "method_label": METHOD_LABELS.get(payload["method"], payload["method"]),
                "window_index": int(payload["window_index"]),
                "train_end": payload["train_end"],
                "seed": int(payload["seed"]),
                "terminal_objective": float(payload["optimizer"]["objective"]),
                "budget_mean": budget.get("mean"),
                "budget_std": budget.get("std"),
                "budget_q25": budget.get("q25"),
                "budget_median": budget.get("median"),
                "budget_q75": budget.get("q75"),
                "p_explore_mean": probability.get("mean"),
                "p_explore_std": probability.get("std"),
                "p_explore_q25": probability.get("q25"),
                "p_explore_median": probability.get("median"),
                "p_explore_q75": probability.get("q75"),
                "heavy_explore_count": int(counts["heavy_explore"]),
                "conservative_count": int(counts["conservative"]),
                "governed_proposals": governed,
                "realized_heavy_explore_rate": (
                    int(counts["heavy_explore"]) / governed if governed else np.nan
                ),
                "budget_p_explore_correlation": diagnostics.get(
                    "budget_p_explore_correlation"
                ),
            }
        )
    runs = pd.DataFrame.from_records(rows).sort_values(
        ["method", "window_index", "seed"]
    )
    selected = runs[runs["method"].isin(["rsf_abc", "fixed_mix_abc"])].copy()
    if selected.empty:
        raise ValueError("Mechanism records for RSF and Fixed-Mix are required")

    method_rows = []
    for method, group in selected.groupby("method", sort=True):
        total_heavy = int(group["heavy_explore_count"].sum())
        total_governed = int(group["governed_proposals"].sum())
        method_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS[method],
                "diagnostic_runs": len(group),
                "dates": group["train_end"].nunique(),
                "seeds": group["seed"].nunique(),
                "mean_terminal_objective": float(group["terminal_objective"].mean()),
                "median_terminal_objective": float(group["terminal_objective"].median()),
                "mean_budget": float(group["budget_mean"].mean()),
                "mean_p_explore": float(group["p_explore_mean"].mean()),
                "realized_heavy_explore_rate": total_heavy / total_governed,
                "mean_budget_p_explore_correlation": (
                    float(group["budget_p_explore_correlation"].dropna().mean())
                    if group["budget_p_explore_correlation"].notna().any()
                    else np.nan
                ),
            }
        )
    methods = pd.DataFrame.from_records(method_rows)

    dates = (
        selected.groupby(["method", "method_label", "window_index", "train_end"], as_index=False)
        .agg(
            mean_terminal_objective=("terminal_objective", "mean"),
            median_terminal_objective=("terminal_objective", "median"),
            mean_budget=("budget_mean", "mean"),
            mean_p_explore=("p_explore_mean", "mean"),
            mean_realized_heavy_rate=("realized_heavy_explore_rate", "mean"),
            seeds=("seed", "nunique"),
        )
        .sort_values(["method", "window_index"])
    )

    pivot = dates.pivot(
        index=["window_index", "train_end"],
        columns="method",
        values="mean_terminal_objective",
    ).reset_index()
    pivot["rsf_minus_fixed_mix"] = pivot["rsf_abc"] - pivot["fixed_mix_abc"]
    w_plus, p_value, rank_biserial = exact_paired_wilcoxon(
        pivot["rsf_minus_fixed_mix"].to_numpy(dtype=float)
    )
    test = pd.DataFrame.from_records(
        [
            {
                "comparison": "rsf_abc_minus_fixed_mix_abc",
                "date_level_estimand": "mean terminal formation objective across paired seeds",
                "n_dates": len(pivot),
                "mean_date_level_difference": float(pivot["rsf_minus_fixed_mix"].mean()),
                "median_date_level_difference": float(pivot["rsf_minus_fixed_mix"].median()),
                "wilcoxon_w_plus": w_plus,
                "exact_two_sided_p": p_value,
                "matched_rank_biserial": rank_biserial,
            }
        ]
    )
    return runs, methods, dates, pivot, test


def run_g5_analysis(
    project_root: Path, config: dict[str, Any], config_path: Path
) -> tuple[Path, dict[str, Any]]:
    start = time.perf_counter()
    settings = config["analysis"]
    batch_id = str(settings["batch_id"])
    batch_root = project_root / "results" / "batch" / batch_id
    output_root = project_root / str(settings["output_dir"])
    table_dir = output_root / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)

    main_paths = _active_json_paths(batch_root / "main")
    diagnostic_paths = _active_json_paths(batch_root / "diagnostics")
    main_payloads = []
    for path in main_paths:
        payload = _load_json(path)
        payload["source_path"] = str(path)
        main_payloads.append(payload)
    diagnostic_payloads = [_load_json(path) for path in diagnostic_paths]
    required_methods = {
        str(settings["primary_method"]), *[str(x) for x in settings["comparators"]]
    }
    observed_methods = {str(payload["method"]) for payload in main_payloads}
    if not required_methods.issubset(observed_methods):
        raise ValueError(
            f"Missing main methods: {sorted(required_methods - observed_methods)}"
        )

    main_paths_table = _main_path_table(main_payloads)
    main_summary = (
        main_paths_table.groupby(["method", "method_label"], as_index=False)
        .agg(
            paths=("seed", "count"),
            mean_net_omega=("omega_zero_threshold", "mean"),
            sd_net_omega=("omega_zero_threshold", "std"),
            mean_annualized_return=("annualized_return", "mean"),
            mean_annualized_volatility=("annualized_volatility", "mean"),
            mean_sharpe=("sharpe_zero_rf", "mean"),
            mean_sortino=("sortino_zero_threshold", "mean"),
            mean_max_drawdown=("max_drawdown", "mean"),
            mean_daily_cvar95=("cvar95_daily_loss", "mean"),
            mean_monthly_turnover=("mean_monthly_turnover", "mean"),
        )
        .sort_values("method")
    )
    seed_effects, bootstrap = hierarchical_paired_bootstrap(
        main_payloads,
        str(settings["primary_method"]),
        [str(value) for value in settings["comparators"]],
        int(settings["bootstrap_replicates"]),
        int(settings["moving_block_length"]),
        int(settings["random_seed"]),
    )
    (
        mechanism_runs,
        mechanism_methods,
        mechanism_dates,
        mechanism_date_effects,
        mechanism_pairwise_test,
    ) = (
        _mechanism_tables(diagnostic_payloads)
    )

    tables = {
        "main_path_metrics": main_paths_table,
        "main_method_summary": main_summary,
        "paired_seed_effects": seed_effects,
        "hierarchical_paired_bootstrap": bootstrap,
        "mechanism_run_diagnostics": mechanism_runs,
        "mechanism_method_summary": mechanism_methods,
        "mechanism_date_summary": mechanism_dates,
        "mechanism_date_effects": mechanism_date_effects,
        "mechanism_pairwise_test": mechanism_pairwise_test,
    }
    for name, frame in tables.items():
        frame.to_csv(table_dir / f"{name}.csv", index=False)

    calibration_path = batch_root / "g5_fixed_mix_calibration.json"
    manifest = {
        "analysis_version": config["project"]["analysis_version"],
        "batch_id": batch_id,
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "calibration_path": str(calibration_path),
        "calibration_sha256": (
            sha256_file(calibration_path) if calibration_path.exists() else None
        ),
        "raw_counts": {
            "main_paths": len(main_payloads),
            "diagnostic_runs": len(diagnostic_payloads),
        },
        "raw_sha256": {
            str(path.relative_to(project_root)): sha256_file(path)
            for path in [*main_paths, *diagnostic_paths]
        },
        "table_sha256": {
            path.name: sha256_file(path) for path in sorted(table_dir.glob("*.csv"))
        },
        "environment": environment_record(),
        "wall_seconds": time.perf_counter() - start,
    }
    manifest_path = output_root / "g5_analysis_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, manifest
