from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pytest

from rsfabc_portfolio.data import parse_french_industry_zip


def test_value_weighted_parser_with_synthetic_archive(tmp_path):
    archive = tmp_path / "2_Industry_Portfolios_daily_CSV.zip"
    payload = """Synthetic fixture
Average Value Weighted Returns -- Daily
,Industry A,Industry B
20060103,1.00,-2.00
20060104,0.50,1.50

Average Equal Weighted Returns -- Daily
,Industry A,Industry B
20060103,1.10,-1.90
"""
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        bundle.writestr("2_Industry_Portfolios_daily.CSV", payload)

    frame, snapshot = parse_french_industry_zip(
        archive,
        universe=2,
        start_date="2006-01-01",
        end_date="2006-01-31",
        weighting="value",
    )

    assert frame.shape == (2, 2)
    assert frame.iloc[0].tolist() == pytest.approx([0.01, -0.02])
    assert snapshot.rows_final == 2


@pytest.mark.skipif(
    not Path("data/raw/49_Industry_Portfolios_daily_CSV.zip").exists(),
    reason="official archive is downloaded by the optional integration workflow",
)
def test_official_49_industry_value_weighted_parser():
    archive = Path("data/raw/49_Industry_Portfolios_daily_CSV.zip")
    frame, snapshot = parse_french_industry_zip(
        archive,
        universe=49,
        start_date="2006-01-01",
        end_date="2025-12-31",
        weighting="value",
    )
    assert frame.shape[1] == 49
    assert len(frame) > 4500
    assert frame.index[0].year == 2006
    assert frame.index[-1].date().isoformat() == "2025-12-31"
    assert len(snapshot.archive_sha256) == 64
    assert not frame.isna().any().any()
    assert frame.abs().max().max() < 1.0
