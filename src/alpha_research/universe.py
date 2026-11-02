"""Eligibility uses only observations through the signal close."""


def select_universe(panel, settings):
    frame = panel.copy()
    observed = frame.close.notna() & frame.volume.gt(0)
    frame["history_sessions"] = observed.groupby(frame.security_id).cumsum()
    frame["liquidity"] = frame.groupby("security_id").value.transform(
        lambda s: s.rolling(settings["liquidity_window"], min_periods=settings["liquidity_window"]).mean())
    frame["eligible"] = (frame.security_type.isin(["common", "common_candidate"]) &
                         frame.history_sessions.ge(settings["min_history"]) & observed &
                         frame.liquidity.gt(0))
    ranked = frame.loc[frame.eligible].sort_values(
        ["date", "liquidity", "security_id"], ascending=[True, False, True])
    selected = ranked.groupby("date").head(settings["universe_size"]).index
    frame["selected"] = frame.index.isin(selected)
    return frame
