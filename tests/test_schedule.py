import numpy as np
import pandas as pd

from rsfabc_portfolio.schedule import build_monthly_schedule, evenly_spaced_window_indices


def test_schedule_uses_exact_trailing_window_without_overlap():
    index = pd.bdate_range("2006-01-02", periods=900)
    windows = build_monthly_schedule(index, train_days=504)
    first = windows[0]
    assert len(first.train_positions) == 504
    assert first.train_positions[-1] + 1 == first.hold_positions[0]
    assert pd.Timestamp(first.train_end) < pd.Timestamp(first.hold_start)
    assert all(
        window.hold_positions[0] == window.train_positions[-1] + 1 for window in windows
    )


def test_evenly_spaced_indices_are_unique_and_include_edges():
    indices = evenly_spaced_window_indices(216, 12)
    assert len(indices) == 12
    assert indices[0] == 0
    assert indices[-1] == 215
    assert np.all(np.diff(indices) > 0)
