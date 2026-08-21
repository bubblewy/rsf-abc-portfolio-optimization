"""Command-line entrypoint for data preparation and experiment execution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config, project_path
from .batch import run_batch_phase
from .data import load_processed_dataset, prepare_all
from .runner import run_runtime_pilot, run_walk_forward_path
from .schedule import build_monthly_schedule
from .provenance import utc_run_id, write_json


def _processed_path(project_root: Path, config: dict, universe: int) -> Path:
    data_config = config["data"]
    start = data_config["start_date"]
    end = data_config["end_date"]
    directory = project_path(project_root, data_config["processed_dir"])
    return directory / f"industry_{universe}_daily_{start}_{end}.csv.gz"


def command_prepare_data(config_path: Path) -> None:
    config, project_root = load_config(config_path)
    data_config = config["data"]
    outputs = prepare_all(
        project_path(project_root, data_config["raw_dir"]),
        project_path(project_root, data_config["processed_dir"]),
        [int(value) for value in data_config["universes"]],
        data_config["start_date"],
        data_config["end_date"],
        data_config["weighting"],
    )
    print(json.dumps(outputs, indent=2))


def command_schedule(config_path: Path) -> None:
    config, project_root = load_config(config_path)
    universe = int(config["data"]["primary_universe"])
    path = _processed_path(project_root, config, universe)
    frame = load_processed_dataset(path)
    windows = build_monthly_schedule(frame.index, int(config["walk_forward"]["train_days"]))
    payload = {
        "windows": len(windows),
        "first": {
            "train_start": windows[0].train_start,
            "train_end": windows[0].train_end,
            "hold_start": windows[0].hold_start,
            "hold_end": windows[0].hold_end,
        },
        "last": {
            "train_start": windows[-1].train_start,
            "train_end": windows[-1].train_end,
            "hold_start": windows[-1].hold_start,
            "hold_end": windows[-1].hold_end,
        },
    }
    print(json.dumps(payload, indent=2))


def command_pilot(config_path: Path) -> None:
    config, project_root = load_config(config_path)
    universe = int(config["data"]["primary_universe"])
    path = _processed_path(project_root, config, universe)
    frame = load_processed_dataset(path)
    output_path, payload = run_runtime_pilot(
        frame,
        config,
        path,
        project_root / "results" / "pilot",
    )
    print(json.dumps({"output": str(output_path), "projection": payload["projection"]}, indent=2))


def command_run_path(config_path: Path, method: str, seed: int) -> None:
    config, project_root = load_config(config_path)
    universe = int(config["data"]["primary_universe"])
    path = _processed_path(project_root, config, universe)
    frame = load_processed_dataset(path)
    payload = run_walk_forward_path(
        frame,
        method,
        seed,
        config["portfolio"],
        config["optimizers"],
        int(config["walk_forward"]["train_days"]),
    )
    payload["run_id"] = utc_run_id(f"path_{method}_{seed}")
    payload["data_path"] = str(path)
    output = project_root / "results" / "raw" / f"{payload['run_id']}.json"
    write_json(output, payload)
    print(json.dumps({"output": str(output), "rebalances": len(payload["rebalances"])}, indent=2))


def command_run_batch(config_path: Path, phase: str, batch_id: str) -> None:
    config, project_root = load_config(config_path)
    summary_path, summary = run_batch_phase(
        config, project_root, phase, batch_id, config_path.resolve()
    )
    print(json.dumps({"summary_path": str(summary_path), "summary": summary}, indent=2))


def command_analyze(config_path: Path) -> None:
    from .analysis import run_confirmatory_analysis

    config, project_root = load_config(config_path)
    manifest_path, manifest = run_confirmatory_analysis(
        project_root, config, config_path.resolve()
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "raw_counts": manifest["raw_counts"],
                "experiment_matrix_rows": manifest["experiment_matrix_rows"],
                "wall_seconds": manifest["wall_seconds"],
            },
            indent=2,
        )
    )


def command_calibrate_fixed_mix(config_path: Path, batch_id: str) -> None:
    from .g5_analysis import calibrate_fixed_mix

    _, project_root = load_config(config_path)
    output, record = calibrate_fixed_mix(project_root, batch_id)
    print(
        json.dumps(
            {
                "output": str(output),
                "fixed_explore_probability": record["fixed_explore_probability"],
                "diagnostic_runs": record["diagnostic_runs"],
            },
            indent=2,
        )
    )


def command_analyze_g5(config_path: Path) -> None:
    from .g5_analysis import run_g5_analysis

    config, project_root = load_config(config_path)
    manifest_path, manifest = run_g5_analysis(
        project_root, config, config_path.resolve()
    )
    print(
        json.dumps(
            {
                "manifest_path": str(manifest_path),
                "raw_counts": manifest["raw_counts"],
                "wall_seconds": manifest["wall_seconds"],
            },
            indent=2,
        )
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rsfabc")
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare-data", "schedule", "pilot", "analyze", "analyze-g5"):
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, required=True)
    calibration_parser = subparsers.add_parser("calibrate-fixed-mix")
    calibration_parser.add_argument("--config", type=Path, required=True)
    calibration_parser.add_argument("--batch-id", required=True)
    path_parser = subparsers.add_parser("run-path")
    path_parser.add_argument("--config", type=Path, required=True)
    path_parser.add_argument("--method", required=True)
    path_parser.add_argument("--seed", type=int, required=True)
    batch_parser = subparsers.add_parser("run-batch")
    batch_parser.add_argument("--config", type=Path, required=True)
    batch_parser.add_argument(
        "--phase", choices=["main", "diagnostics", "robustness"], required=True
    )
    batch_parser.add_argument("--batch-id", required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "prepare-data":
        command_prepare_data(args.config)
    elif args.command == "schedule":
        command_schedule(args.config)
    elif args.command == "pilot":
        command_pilot(args.config)
    elif args.command == "run-path":
        command_run_path(args.config, args.method, args.seed)
    elif args.command == "analyze":
        command_analyze(args.config)
    elif args.command == "analyze-g5":
        command_analyze_g5(args.config)
    elif args.command == "calibrate-fixed-mix":
        command_calibrate_fixed_mix(args.config, args.batch_id)
    else:
        command_run_batch(args.config, args.phase, args.batch_id)


if __name__ == "__main__":
    main()
