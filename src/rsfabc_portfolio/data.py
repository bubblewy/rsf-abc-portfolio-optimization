"""Official Kenneth R. French industry-portfolio data ingestion."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import shutil
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd


FRENCH_URL_TEMPLATE = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "{n}_Industry_Portfolios_daily_CSV.zip"
)
VALUE_MARKER = "Average Value Weighted Returns -- Daily"
EQUAL_MARKER = "Average Equal Weighted Returns -- Daily"
MISSING_CODES = {-99.99, -999.0}


def download_french_archive(universe: int, destination: Path) -> Path:
    """Download one official Kenneth R. French industry archive atomically."""
    destination = Path(destination)
    if destination.exists():
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    source_url = FRENCH_URL_TEMPLATE.format(n=universe)
    request = Request(
        source_url,
        headers={"User-Agent": "rsfabc-portfolio/0.1 (academic reproducibility)"},
    )
    temporary = destination.with_suffix(destination.suffix + ".part")
    try:
        with urlopen(request, timeout=60) as response, temporary.open("wb") as handle:
            shutil.copyfileobj(response, handle)
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


@dataclass(frozen=True)
class DataSnapshot:
    universe: int
    source_url: str
    archive_path: str
    archive_sha256: str
    member_name: str
    weighting: str
    requested_start: str
    requested_end: str
    actual_start: str
    actual_end: str
    rows_before_date_filter: int
    rows_in_date_range: int
    rows_removed_missing: int
    rows_final: int
    columns: list[str]
    created_at_utc: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_section(text: str, marker: str) -> tuple[list[str], list[list[str]]]:
    lines = text.splitlines()
    try:
        marker_index = next(i for i, line in enumerate(lines) if marker in line)
    except StopIteration as exc:
        raise ValueError(f"Section marker not found: {marker}") from exc

    header_index = marker_index + 1
    while header_index < len(lines) and not lines[header_index].strip():
        header_index += 1
    if header_index >= len(lines):
        raise ValueError(f"Header not found after marker: {marker}")

    header = next(csv.reader([lines[header_index]]))
    if not header or header[0].strip() != "":
        raise ValueError("Unexpected French CSV header: first field should be blank date label")
    columns = [item.strip() for item in header[1:]]

    records: list[list[str]] = []
    for line in lines[header_index + 1 :]:
        if not line.strip():
            break
        row = next(csv.reader([line]))
        date_token = row[0].strip() if row else ""
        if len(date_token) != 8 or not date_token.isdigit():
            break
        if len(row) != len(columns) + 1:
            raise ValueError(
                f"Row {date_token} has {len(row) - 1} returns; expected {len(columns)}"
            )
        records.append([item.strip() for item in row])
    if not records:
        raise ValueError(f"No data rows found after marker: {marker}")
    return columns, records


def parse_french_industry_zip(
    archive_path: Path,
    universe: int,
    start_date: str,
    end_date: str,
    weighting: str = "value",
) -> tuple[pd.DataFrame, DataSnapshot]:
    archive_path = Path(archive_path)
    if weighting not in {"value", "equal"}:
        raise ValueError("weighting must be 'value' or 'equal'")
    marker = VALUE_MARKER if weighting == "value" else EQUAL_MARKER

    with zipfile.ZipFile(archive_path) as archive:
        members = [name for name in archive.namelist() if name.lower().endswith(".csv")]
        if len(members) != 1:
            raise ValueError(f"Expected one CSV member, found {members}")
        member_name = members[0]
        text = archive.read(member_name).decode("utf-8-sig")

    columns, records = _extract_section(text, marker)
    if len(columns) != universe:
        raise ValueError(f"Archive declares {len(columns)} industries; expected {universe}")

    raw = pd.DataFrame(records, columns=["date", *columns])
    raw["date"] = pd.to_datetime(raw["date"], format="%Y%m%d", errors="raise")
    for column in columns:
        raw[column] = pd.to_numeric(raw[column], errors="raise")
    rows_before = len(raw)

    dated = raw.loc[
        (raw["date"] >= pd.Timestamp(start_date))
        & (raw["date"] <= pd.Timestamp(end_date))
    ].copy()
    rows_in_range = len(dated)
    dated[columns] = dated[columns].replace(list(MISSING_CODES), np.nan)
    missing_mask = dated[columns].isna().any(axis=1)
    rows_removed = int(missing_mask.sum())
    clean = dated.loc[~missing_mask].set_index("date").sort_index()
    clean = clean.astype(float) / 100.0

    if clean.empty:
        raise ValueError("No complete rows remain after date and missing-value filters")
    if not clean.index.is_monotonic_increasing or not clean.index.is_unique:
        raise ValueError("Parsed dates must be unique and increasing")

    snapshot = DataSnapshot(
        universe=universe,
        source_url=FRENCH_URL_TEMPLATE.format(n=universe),
        archive_path=str(archive_path),
        archive_sha256=sha256_file(archive_path),
        member_name=member_name,
        weighting=weighting,
        requested_start=start_date,
        requested_end=end_date,
        actual_start=clean.index[0].date().isoformat(),
        actual_end=clean.index[-1].date().isoformat(),
        rows_before_date_filter=rows_before,
        rows_in_date_range=rows_in_range,
        rows_removed_missing=rows_removed,
        rows_final=len(clean),
        columns=list(clean.columns),
        created_at_utc=datetime.now(timezone.utc).isoformat(),
    )
    return clean, snapshot


def write_processed_dataset(
    frame: pd.DataFrame,
    snapshot: DataSnapshot,
    processed_dir: Path,
) -> tuple[Path, Path]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    stem = f"industry_{snapshot.universe}_daily_{snapshot.requested_start}_{snapshot.requested_end}"
    data_path = processed_dir / f"{stem}.csv.gz"
    manifest_path = processed_dir / f"{stem}.manifest.json"
    frame.to_csv(data_path, index_label="date", compression="gzip", float_format="%.10g")
    payload = asdict(snapshot)
    payload["processed_path"] = str(data_path)
    payload["processed_sha256"] = sha256_file(data_path)
    manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return data_path, manifest_path


def load_processed_dataset(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, index_col="date", parse_dates=["date"])
    if frame.empty or frame.isna().any().any():
        raise ValueError(f"Processed dataset is empty or contains missing data: {path}")
    if not frame.index.is_monotonic_increasing or not frame.index.is_unique:
        raise ValueError(f"Processed dates are not unique and increasing: {path}")
    return frame.astype(float)


def prepare_all(
    raw_dir: Path,
    processed_dir: Path,
    universes: Iterable[int],
    start_date: str,
    end_date: str,
    weighting: str,
) -> list[dict[str, str]]:
    outputs: list[dict[str, str]] = []
    for universe in universes:
        archive = Path(raw_dir) / f"{universe}_Industry_Portfolios_daily_CSV.zip"
        download_french_archive(universe, archive)
        frame, snapshot = parse_french_industry_zip(
            archive, universe, start_date, end_date, weighting
        )
        data_path, manifest_path = write_processed_dataset(frame, snapshot, processed_dir)
        outputs.append(
            {
                "universe": str(universe),
                "data_path": str(data_path),
                "manifest_path": str(manifest_path),
                "rows": str(len(frame)),
            }
        )
    return outputs
