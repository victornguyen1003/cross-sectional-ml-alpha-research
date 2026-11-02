"""Forward labels retain explicit entry/end dates for chronological purging."""

import numpy as np
import pandas as pd


def signal_dates(calendar, settings):
    dates = pd.DatetimeIndex(calendar)
    dates = dates[dates >= pd.Timestamp(settings["train_start"])]
    return dates[::settings["rebalance"]]


def build_labels(panel, calendar, settings):
    frame = panel.copy()
    group = frame.groupby("security_id", sort=False)
    horizon = settings["horizon"]
    frame["entry_date"] = group.date.shift(-1)
    frame["label_end"] = group.date.shift(-(horizon + 1))
    entry = group.total_open.shift(-1)
    if "open" in frame:
        entry = entry.where(group.open.shift(-1).notna())
    exit_price = group.total_open.shift(-(horizon + 1))
    frame["realized_return"] = (exit_price / entry - 1).replace([np.inf, -np.inf], np.nan)
    # Future values never participate in eligibility or signal-date selection.
    frame = frame[frame.selected & frame.date.isin(signal_dates(calendar, settings))].copy()
    frame["target"] = frame.groupby("date").realized_return.rank(method="average", pct=True)
    frame["label_available"] = frame.target.notna() & frame.label_end.notna()
    return frame.reset_index(drop=True)
