"""Strict no-look-ahead monthly walk-forward schedule."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .types import WalkForwardWindow


def build_monthly_schedule(index: pd.DatetimeIndex, train_days: int) -> list[WalkForwardWindow]:
    if train_days <= 0:
        raise ValueError("train_days must be positive")
    if not index.is_monotonic_increasing or not index.is_unique:
        raise ValueError("index must be unique and increasing")
    periods = index.to_period("M")
    windows: list[WalkForwardWindow] = []
    for period in periods.unique():
        hold_positions = np.flatnonzero(periods == period)
        if hold_positions.size == 0:
            continue
        first = int(hold_positions[0])
        if first < train_days:
            continue
        train_positions = np.arange(first - train_days, first, dtype=int)
        if index[train_positions[-1]] >= index[hold_positions[0]]:
            raise AssertionError("training data overlaps the holding month")
        windows.append(
            WalkForwardWindow(
                index=len(windows),
                train_start=index[train_positions[0]].date().isoformat(),
                train_end=index[train_positions[-1]].date().isoformat(),
                hold_start=index[hold_positions[0]].date().isoformat(),
                hold_end=index[hold_positions[-1]].date().isoformat(),
                train_positions=train_positions,
                hold_positions=hold_positions.astype(int),
            )
        )
    if not windows:
        raise ValueError("no eligible walk-forward windows")
    return windows


def evenly_spaced_window_indices(window_count: int, count: int) -> np.ndarray:
    if window_count <= 0 or count <= 0 or count > window_count:
        raise ValueError("invalid window/count combination")
    return np.unique(np.rint(np.linspace(0, window_count - 1, count)).astype(int))
