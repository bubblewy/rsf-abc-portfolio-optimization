# RSF-ABC Portfolio Optimization

Reproducible research code for **Risk-Sensitive Foraging Artificial Bee Colony
(RSF-ABC)** optimization applied to cardinality-constrained Omega portfolio
rebalancing under transaction costs.

The project implements an explicit biomimetic controller: an Omega-derived
gain-loss budget state controls the probability of selecting heavy-tailed
exploration versus conservative best-guided search. It also includes
Fixed-Mix-ABC, a state-independent matched comparator that separates the value
of state conditioning from the value of the two proposal kernels themselves.

## What is included

- RSF-ABC, Standard ABC, HT-ABC, RS-Light-ABC, and Fixed-Mix-ABC;
- Particle Swarm Optimization and Differential Evolution;
- equal-weight, cardinality-matched equal-weight, minimum-variance, and
  mixed-integer maximum-Omega baselines;
- exact-cardinality repair, active-weight bounds, portfolio drift, and
  proportional transaction costs;
- look-ahead-free monthly walk-forward experiments;
- mechanism diagnostics, ablations, robustness conditions, paired inference,
  and publication-figure generation;
- frozen YAML configurations and automated tests.

Raw market data, generated results, manuscript files, and local environments
are intentionally not committed.

## Quick start

Python 3.11 or newer is required.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e '.[dev]'
.venv/bin/pytest -q
.venv/bin/rsfabc prepare-data --config configs/pilot.yaml
.venv/bin/rsfabc pilot --config configs/pilot.yaml
```

`prepare-data` downloads the official daily 10-, 30-, and 49-industry archives
from the Kenneth R. French Data Library when they are absent, then writes
processed files and provenance manifests under `data/`.

For an exact dependency snapshot instead of the flexible project constraints:

```bash
.venv/bin/python -m pip install -r requirements.lock
.venv/bin/python -m pip install -e . --no-deps
```

## G5 matched-mechanism reproduction

```bash
.venv/bin/rsfabc prepare-data --config configs/g5_rsf_diagnostic_calibration.yaml
.venv/bin/pytest -q
.venv/bin/rsfabc run-batch \
  --config configs/g5_rsf_diagnostic_calibration.yaml \
  --phase diagnostics \
  --batch-id g5_evidence_repair_20260818
.venv/bin/rsfabc calibrate-fixed-mix \
  --config configs/g5_rsf_diagnostic_calibration.yaml \
  --batch-id g5_evidence_repair_20260818
.venv/bin/rsfabc run-batch \
  --config configs/g5_evidence_repair.yaml \
  --phase diagnostics \
  --batch-id g5_evidence_repair_20260818
.venv/bin/rsfabc run-batch \
  --config configs/g5_evidence_repair.yaml \
  --phase main \
  --batch-id g5_evidence_repair_20260818
.venv/bin/rsfabc analyze-g5 --config configs/g5_analysis.yaml
```

The calibration step should return
`fixed_explore_probability = 0.43254216553074837`. Batch records are
append-only, and completed records are skipped when a command is resumed.

## Repository layout

```text
configs/                 frozen experiment and analysis configurations
scripts/                 package-building and validation utilities
src/rsfabc_portfolio/    algorithms, portfolio logic, runners, and analysis
tests/                   offline unit tests and optional data integration test
```

Generated directories are ignored by Git:

```text
data/                    downloaded and processed source data
results/                 raw runs, summaries, tables, and figures
submission/              locally assembled reproducibility archives
```

## Experimental boundary

The repository supports a mechanism-and-boundary study, not a claim that
RSF-ABC is a universally superior optimizer or an executable investment
strategy. Formation objectives and historical backtests are not evidence of
future investment performance.

## Data provenance

The source data are daily value-weighted industry portfolios from the
[Kenneth R. French Data Library](https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html).
The original archives are not redistributed. Each processed-data manifest
records the source URL, archive SHA-256, date range, columns, and output hash.

## License

Code is released under the MIT License. Third-party data remain subject to the
terms of their original provider.
