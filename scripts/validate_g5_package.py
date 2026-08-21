from __future__ import annotations

import csv
import hashlib
import json
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECTS = ROOT.parent
BATCH_ID = "g5_evidence_repair_20260818"
BATCH = ROOT / "results" / "batch" / BATCH_ID
ANALYSIS = ROOT / "results" / "analysis" / BATCH_ID
TABLES = ANALYSIS / "tables"
VALIDATION = ANALYSIS / "validation"
MANUSCRIPT_ROOT = PROJECTS / "rsf_abc_manuscript_optimization_20260818"
FIGURE_ROOT = PROJECTS / "rsf_abc_figures_20260818"
ARCHIVE = ROOT / "submission" / "RSFABC_G5_Evidence_Repair.zip"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def require(condition: bool, message: str, checks: list[dict[str, str]]) -> None:
    if not condition:
        raise AssertionError(message)
    checks.append({"status": "PASS", "check": message})


def main() -> None:
    checks: list[dict[str, str]] = []

    diagnostic_files = sorted((BATCH / "diagnostics").glob("*.json"))
    main_files = sorted((BATCH / "main").glob("*.json"))
    require(len(diagnostic_files) == 1080, "1,080 diagnostic JSON records present", checks)
    require(len(main_files) == 15, "15 complete-path JSON records present", checks)

    main_summary = json.loads((BATCH / "main_summary.json").read_text(encoding="utf-8"))
    diag_summary = json.loads((BATCH / "diagnostics_summary.json").read_text(encoding="utf-8"))
    require(main_summary["failed_tasks"] == 0 and main_summary["optimizer_failures"] == 0, "main batch has zero failures", checks)
    require(diag_summary["failed_tasks"] == 0 and diag_summary["optimizer_failures"] == 0, "diagnostic batch has zero failures", checks)
    require(main_summary["config_sha256"] == "8e0c35ee0d776cb042397ed97fae3bb3e4125ad7f65b7411f36d30c93c5f0c0d", "final G5 configuration hash matches frozen identity", checks)

    calibration = json.loads((BATCH / "g5_fixed_mix_calibration.json").read_text(encoding="utf-8"))
    require(calibration["heavy_explore_proposals"] + calibration["conservative_proposals"] == calibration["governed_proposals"], "calibration branch counts sum exactly", checks)
    require(abs(calibration["fixed_explore_probability"] - 0.43254216553074837) < 1e-15, "Fixed-Mix probability matches pooled RSF realized rate", checks)
    require(len(calibration["diagnostic_dates"]) == 12 and calibration["diagnostic_runs"] == 360, "calibration uses 12 dates and 360 RSF runs", checks)

    manifest = json.loads((ANALYSIS / "g5_analysis_manifest.json").read_text(encoding="utf-8"))
    require(manifest["raw_counts"] == {"diagnostic_runs": 1080, "main_paths": 15}, "analysis manifest raw counts match files", checks)
    require(len(manifest["raw_sha256"]) == 1095, "analysis manifest hashes every G5 raw record", checks)

    method_rows = {row["method"]: row for row in read_csv("main_method_summary.csv")}
    expected_omega = {
        "rsf_abc": 1.1090489206156957,
        "standard_abc": 1.1096983287120885,
        "fixed_mix_abc": 1.1099839921703483,
    }
    for method, expected in expected_omega.items():
        require(abs(float(method_rows[method]["mean_net_omega"]) - expected) < 1e-15, f"{method} mean net Omega matches frozen result", checks)

    bootstrap = {row["comparator"]: row for row in read_csv("hierarchical_paired_bootstrap.csv")}
    require(abs(float(bootstrap["standard_abc"]["estimate_primary_minus_comparator"]) + 0.0006494080963925786) < 1e-15, "RSF-minus-Standard point effect matches complete paths", checks)
    require(abs(float(bootstrap["fixed_mix_abc"]["estimate_primary_minus_comparator"]) + 0.0009350715546524224) < 1e-15, "RSF-minus-Fixed-Mix point effect matches complete paths", checks)
    require(all(float(row["ci_lower_95"]) < 0 < float(row["ci_upper_95"]) for row in bootstrap.values()), "both G5 intervals span zero", checks)
    require(len(read_csv("paired_seed_effects.csv")) == 10, "all ten comparator-by-seed effects are exported", checks)
    require(len(read_csv("mechanism_date_effects.csv")) == 12, "all 12 date-level Fixed-Mix effects are exported", checks)

    raw_hashes = {
        10: "c4c9b7c3969e5477630e9e312f61a19bb74c0f8d159612ef3adf33380eeb301c",
        30: "ca3da11ca28809eb4edf3b035a17a89ace589d1e5f180c20a7e85fb36c18bff8",
        49: "9b368230566fbf636b64b7a766a74b9a94a15dda5bb9101ff141670526dc1a34",
    }
    for universe, expected in raw_hashes.items():
        path = ROOT / "data" / "raw" / f"{universe}_Industry_Portfolios_daily_CSV.zip"
        require(sha256(path) == expected, f"{universe}-industry raw ZIP identity matches recorded source", checks)

    legacy_rows = {}
    with (FIGURE_ROOT / "data" / "main_results.csv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            legacy_rows[row["method"]] = row
    require(round(float(method_rows["rsf_abc"]["mean_net_omega"]), 4) == float(legacy_rows["RSF-ABC"]["net_omega"]), "G5 RSF mean reproduces legacy Figure 3 value", checks)
    require(round(float(method_rows["standard_abc"]["mean_net_omega"]), 4) == float(legacy_rows["Standard ABC"]["net_omega"]), "G5 Standard mean reproduces legacy Figure 3 value", checks)

    manuscript = MANUSCRIPT_ROOT / "docs" / "G5_revised_manuscript_round2.md"
    supplement = MANUSCRIPT_ROOT / "docs" / "G5_revised_supplement.md"
    manuscript_text = manuscript.read_text(encoding="utf-8")
    supplement_text = supplement.read_text(encoding="utf-8")
    for token in ["−0.000649", "−0.016306", "0.015040", "−0.000935", "−0.015859", "0.014490", "−0.99995", "0.43314"]:
        require(token in manuscript_text, f"manuscript contains G5 token {token}", checks)
    require(
        any(
            phrase in manuscript_text
            for phrase in [
                "do not establish equivalence",
                "does not establish equivalence",
                "neither establish equivalence",
            ]
        ),
        "manuscript preserves non-equivalence boundary",
        checks,
    )
    require("Table S1b" in supplement_text and "Table S1d" in supplement_text, "supplement discloses seed and date effects", checks)
    require(not any(byte < 32 and byte not in (9, 10) for byte in manuscript.read_bytes()), "manuscript contains no unexpected control bytes", checks)
    require(not any(byte < 32 and byte not in (9, 10) for byte in supplement.read_bytes()), "supplement contains no unexpected control bytes", checks)

    report1 = json.loads((MANUSCRIPT_ROOT / "docs" / "G5_revision_apply_report.json").read_text(encoding="utf-8"))
    report2 = json.loads((MANUSCRIPT_ROOT / "docs" / "G5_revision_apply_report_round2.json").read_text(encoding="utf-8"))
    require(report1["output_draft_hash"] == report2["base_draft_hash"], "manuscript patch reports form an unbroken chain", checks)
    require(report2["output_draft_hash"] == sha256(manuscript)[:12], "final manuscript hash matches final apply report", checks)
    supplement_report = json.loads((MANUSCRIPT_ROOT / "docs" / "G5_supplement_revision_apply_report.json").read_text(encoding="utf-8"))
    require(supplement_report["output_draft_hash"] == sha256(supplement)[:12], "supplement hash matches apply report", checks)

    svg = FIGURE_ROOT / "figures" / "Figure_3_G5_primary_control_evidence.svg"
    png = FIGURE_ROOT / "previews" / "Figure_3_G5_primary_control_evidence_600dpi.png"
    require(svg.is_file() and png.is_file(), "G5 Figure 3 SVG and 600-dpi preview exist", checks)
    svg_text = svg.read_text(encoding="utf-8")
    require("Palatino Linotype" in svg_text and "−0.99995" in svg_text, "G5 Figure 3 uses requested typography and tracked mechanism statistic", checks)

    archive_hash = sha256(ARCHIVE)
    require(archive_hash == "c33a3f108d09488348354dede447afba19ce0578406c1b8a81aa8e0534a77371", "outer reproduction archive checksum matches manuscript", checks)
    with zipfile.ZipFile(ARCHIVE) as archive:
        require(archive.testzip() is None, "reproduction ZIP passes CRC test", checks)
        require(len(archive.infolist()) == 1159, "reproduction ZIP contains 1,159 entries", checks)

    result = {
        "status": "PASS",
        "checks_passed": len(checks),
        "batch_id": BATCH_ID,
        "archive_sha256": archive_hash,
        "manuscript_sha256": sha256(manuscript),
        "supplement_sha256": sha256(supplement),
        "figure_svg_sha256": sha256(svg),
        "checks": checks,
    }
    VALIDATION.mkdir(parents=True, exist_ok=True)
    target = VALIDATION / "g5_validation.json"
    target.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ["status", "checks_passed", "archive_sha256", "manuscript_sha256", "supplement_sha256", "figure_svg_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
