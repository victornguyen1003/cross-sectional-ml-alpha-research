"""Backward-looking features on a complete exchange-session panel."""

import numpy as np

FEATURES = ["return_5", "return_20", "return_60", "return_120", "return_252",
            "momentum", "volatility_20", "volatility_60", "log_value_20", "relative_volume_20"]


def build_features(panel):
    frame = panel.copy()
    group = frame.groupby("security_id", sort=False)
    for window in [5, 20, 60, 120, 252]:
        frame[f"return_{window}"] = frame.total_close / group.total_close.shift(window) - 1
    frame["momentum"] = group.total_close.shift(20) / group.total_close.shift(252) - 1
    daily = frame.total_close / group.total_close.shift() - 1
    for window in [20, 60]:
        frame[f"volatility_{window}"] = daily.groupby(frame.security_id).transform(
            lambda s: s.rolling(window, min_periods=window).std()) * np.sqrt(252)
    mean_value = group.value.transform(lambda s: s.rolling(20, min_periods=20).mean())
    mean_volume = group.volume.transform(lambda s: s.rolling(20, min_periods=20).mean())
    frame["log_value_20"] = np.log1p(mean_value)
    frame["relative_volume_20"] = frame.volume / mean_volume.where(mean_volume > 0)
    frame[FEATURES] = frame[FEATURES].replace([np.inf, -np.inf], np.nan)
    return frame
