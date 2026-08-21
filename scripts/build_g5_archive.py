from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "g5_evidence_repair_20260818"
PACKAGE_NAME = "RSFABC_G5_Evidence_Repair"
OUTPUT_DIR = ROOT / "submission"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def copy_tree(source: Path, destination: Path) -> None:
    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        if (
            "__pycache__" in path.parts
            or "superseded" in path.parts
            or path.suffix in {".pyc", ".pyo"}
        ):
            continue
        copy_file(path, destination / path.relative_to(source))


def write_manifest(package_root: Path) -> Path:
    entries = []
    for path in sorted(package_root.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            rel = path.relative_to(package_root).as_posix()
            entries.append(f"{sha256(path)}  {rel}")
    manifest = package_root / "MANIFEST.sha256"
    manifest.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return manifest


def build_zip(package_root: Path, target: Path) -> None:
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in sorted(package_root.rglob("*")):
            if not path.is_file():
                continue
            rel = Path(PACKAGE_NAME) / path.relative_to(package_root)
            info = zipfile.ZipInfo(rel.as_posix(), date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o100644 << 16
            archive.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED, compresslevel=9)


def verify_package(package_root: Path, manifest_path: Path) -> dict[str, int]:
    checked = 0
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        expected, rel = line.split("  ", 1)
        path = package_root / rel
        if not path.is_file() or sha256(path) != expected:
            raise RuntimeError(f"checksum verification failed: {rel}")
        checked += 1
    return {
        "manifest_files": checked,
        "diagnostic_json": len(list((package_root / "results" / "batch" / BATCH_ID / "diagnostics").glob("*.json"))),
        "main_json": len(list((package_root / "results" / "batch" / BATCH_ID / "main").glob("*.json"))),
        "analysis_csv": len(list((package_root / "results" / "analysis" / BATCH_ID / "tables").glob("*.csv"))),
    }


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    target = OUTPUT_DIR / f"{PACKAGE_NAME}.zip"
    checksum_path = OUTPUT_DIR / f"{PACKAGE_NAME}.zip.sha256"

    with tempfile.TemporaryDirectory(prefix="rsfabc_g5_archive_") as temp_dir:
        package_root = Path(temp_dir) / PACKAGE_NAME
        package_root.mkdir(parents=True)

        for name in ["README.md", "G5_REPRODUCTION_README.md", "LICENSE", "pyproject.toml", "requirements.lock"]:
            copy_file(ROOT / name, package_root / name)

        copy_tree(ROOT / "src", package_root / "src")
        copy_tree(ROOT / "tests", package_root / "tests")
        copy_tree(ROOT / "configs", package_root / "configs")
        for manifest in sorted((ROOT / "data" / "processed").glob("*.manifest.json")):
            copy_file(manifest, package_root / "data" / "processed" / manifest.name)
        copy_tree(ROOT / "results" / "batch" / BATCH_ID, package_root / "results" / "batch" / BATCH_ID)
        copy_tree(ROOT / "results" / "analysis" / BATCH_ID, package_root / "results" / "analysis" / BATCH_ID)

        manifest_path = write_manifest(package_root)
        counts = verify_package(package_root, manifest_path)
        if counts["diagnostic_json"] != 1080 or counts["main_json"] != 15:
            raise RuntimeError(f"unexpected raw record counts: {counts}")

        build_zip(package_root, target)

    archive_hash = sha256(target)
    checksum_path.write_text(f"{archive_hash}  {target.name}\n", encoding="utf-8")
    with zipfile.ZipFile(target) as archive:
        bad = archive.testzip()
        if bad is not None:
            raise RuntimeError(f"ZIP CRC verification failed: {bad}")
        zip_entries = len(archive.infolist())

    print(json.dumps({
        "archive": str(target),
        "archive_sha256": archive_hash,
        "zip_entries": zip_entries,
        **counts,
    }, indent=2))


if __name__ == "__main__":
    main()
