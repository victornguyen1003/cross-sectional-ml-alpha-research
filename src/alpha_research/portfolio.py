"""Deterministic equal-weight targets; ties break by security identifier."""

import math

import pandas as pd


def target_weights(predictions, decile=0.1, long_only=False):
    rows = []
    for date, frame in predictions.groupby("date", sort=True):
        frame = frame.dropna(subset=["prediction"]).sort_values(["prediction", "security_id"])
        size = math.floor(len(frame) * decile)
        if size < 1 or (not long_only and len(frame) < 2 * size):
            continue
        if long_only:
            for row in frame.itertuples():
                rows.append({"date": date, "entry_date": row.entry_date,
                             "security_id": row.security_id, "weight": 1 / len(frame)})
        else:
            for side, subset in [(-1, frame.head(size)), (1, frame.tail(size))]:
                for row in subset.itertuples():
                    rows.append({"date": date, "entry_date": row.entry_date,
                                 "security_id": row.security_id, "weight": side * 0.5 / size})
    return pd.DataFrame(rows, columns=["date", "entry_date", "security_id", "weight"])
