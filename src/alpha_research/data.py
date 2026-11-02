"""Normalize observed history without substituting today's security universe."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import read_response

PRICE_COLUMNS = ["date", "security_id", "open", "high", "low", "close", "volume", "value"]
ACTION_COLUMNS = ["date", "security_id", "cash_dividend", "share_multiplier",
                  "subscription_ratio", "subscription_price", "payment_date", "terminal_value"]


def number(value):
    text = re.sub(r"<[^>]*>", "", str(value)).replace(",", "").strip()
    return pd.to_numeric(text, errors="coerce")


def parse_date(value):
    text = str(value).strip()
    if re.match(r"^\d{2,3}/\d{2}/\d{2}$", text):
        year, month, day = map(int, text.split("/"))
        return pd.Timestamp(year + 1911, month, day)
    if re.match(r"^\d{8}$", text):
        return pd.to_datetime(text, format="%Y%m%d")
    return pd.Timestamp(text)


def parse_prices(payload, date):
    if "prices" in payload:  # Canonical offline/import fixture format.
        return pd.DataFrame(payload["prices"])
    for table in payload.get("tables", []):
        fields = table.get("fields", [])
        if "Opening Price" not in fields or "Closing Price" not in fields:
            continue
        rows = pd.DataFrame(table["data"], columns=fields)
        aliases = {"Code": "security_id", "Security Code": "security_id", "證券代號": "security_id",
                   "Opening Price": "open", "Highest Price": "high",
                   "Lowest Price": "low", "Closing Price": "close",
                   "Trade Volume": "volume", "Trade Value": "value"}
        rows = rows.rename(columns=aliases)
        if not set(PRICE_COLUMNS[1:]).issubset(rows):
            raise ValueError(f"Unrecognized price fields: {fields}")
        rows["date"] = parse_date(date)
        for column in PRICE_COLUMNS[2:]:
            rows[column] = rows[column].map(number)
        return rows[PRICE_COLUMNS]
    if payload.get("stat") == "OK":
        raise ValueError("TWSE returned OK but no recognized daily-price table")
    return pd.DataFrame(columns=PRICE_COLUMNS)


def parse_actions(payload):
    if "actions" in payload:
        return pd.DataFrame(payload["actions"], columns=ACTION_COLUMNS)
    fields = payload.get("fields", [])
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    # Positional guessing would silently corrupt returns when TWSE changes schema.
    def field(*names):
        for name in names:
            if name in fields:
                return fields.index(name)
        raise ValueError(f"Unrecognized corporate-action fields: {fields}")
    date_i = field("Date", "Ex-right/Ex-dividend Date", "除權息日期")
    id_i = field("Code", "Stock Code", "股票代號")
    cash_i = field("Cash Dividend", "Cash Dividend Per Share", "現金股利")
    kind_i = field("Type", "權/息")
    result = []
    for row in rows:
        # The public reference-price table does not reliably expose all share and
        # subscription terms. Cash-only events are supported; others remain flagged.
        kind = str(row[kind_i]).strip().lower()
        if kind not in {"息", "dividend", "ex-dividend", "xd"}:
            raise ValueError("Non-cash action requires canonical action import with explicit terms")
        result.append({"date": parse_date(row[date_i]), "security_id": str(row[id_i]),
                       "cash_dividend": number(row[cash_i]), "share_multiplier": 1.0,
                       "subscription_ratio": 0.0, "subscription_price": 0.0,
                       "payment_date": pd.NaT, "terminal_value": np.nan})
    return pd.DataFrame(result, columns=ACTION_COLUMNS)


def normalize_frames(prices, actions=None, securities=None):
    prices = prices.copy()
    missing = set(PRICE_COLUMNS) - set(prices)
    if missing:
        raise ValueError(f"Missing price columns: {sorted(missing)}")
    prices["date"] = prices["date"].map(parse_date)
    prices["security_id"] = prices["security_id"].astype(str).str.strip()
    for column in PRICE_COLUMNS[2:]:
        prices[column] = prices[column].map(number)
    if np.isinf(prices[PRICE_COLUMNS[2:]].to_numpy(dtype=float)).any():
        raise ValueError("Infinite market observation")
    if prices.duplicated(["date", "security_id"]).any():
        raise ValueError("Duplicate price keys")
    if prices[["date", "security_id"]].isna().any().any() or prices.security_id.eq("").any():
        raise ValueError("Missing price key")
    if (prices[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Non-positive price")
    if (prices[["volume", "value"]] < 0).any().any():
        raise ValueError("Negative volume/value")
    if ((prices.high < prices.low) | (prices.high < prices[["open", "close"]].max(axis=1)) |
            (prices.low > prices[["open", "close"]].min(axis=1))).any():
        raise ValueError("Inconsistent OHLC prices")
    for column in ["buy_blocked", "sell_blocked"]:
        if column not in prices:
            prices[column] = False
        if not prices[column].isin([True, False]).all():
            raise ValueError(f"{column} must contain booleans")
    # Four-digit stock-code candidates are explicitly UNVERIFIED, never called a
    # point-in-time security master. A dated master overrides the candidate filter.
    prices["security_type"] = np.where(
        prices.security_id.str.fullmatch(r"(?!00|91)\d{4}"), "common_candidate", "other")
    prices["identity_verified"] = False
    if securities is not None:
        securities = securities.copy()
        securities["security_id"] = securities.security_id.astype(str)
        for column in ["valid_from", "valid_to"]:
            securities[column] = pd.to_datetime(securities[column])
        if securities[["valid_from", "valid_to", "security_id", "security_type"]].isna().any().any():
            raise ValueError("Missing security identity terms")
        for sid, group in securities.groupby("security_id"):
            group = group.sort_values("valid_from")
            if (group.valid_from > group.valid_to).any() or (
                    group.valid_from.iloc[1:].to_numpy() <= group.valid_to.iloc[:-1].to_numpy()).any():
                raise ValueError(f"Overlapping/invalid security identity intervals: {sid}")
        for row in securities.itertuples():
            mask = prices.security_id.eq(row.security_id) & prices.date.between(row.valid_from, row.valid_to)
            prices.loc[mask, "security_type"] = row.security_type
            prices.loc[mask, "identity_verified"] = True
    actions = pd.DataFrame(columns=ACTION_COLUMNS) if actions is None else actions.copy()
    defaults = {"cash_dividend": 0.0, "share_multiplier": 1.0, "subscription_ratio": 0.0,
                "subscription_price": 0.0, "payment_date": pd.NaT, "terminal_value": np.nan}
    for column, default in defaults.items():
        if column not in actions:
            actions[column] = default
    actions["date"] = pd.to_datetime(actions.date)
    actions["payment_date"] = pd.to_datetime(actions.payment_date)
    actions["security_id"] = actions.security_id.astype(str)
    if actions.date.isna().any() or actions.security_id.eq("").any():
        raise ValueError("Missing corporate-action key")
    if actions.duplicated(["date", "security_id"]).any():
        raise ValueError("Duplicate corporate-action keys; combine same-day terms explicitly")
    for column in ["cash_dividend", "share_multiplier", "subscription_ratio", "subscription_price"]:
        actions[column] = pd.to_numeric(actions[column], errors="raise").astype(float)
        if not np.isfinite(actions[column]).all() or (actions[column] < 0).any():
            raise ValueError(f"Invalid action term {column}")
    if (actions.share_multiplier <= 0).any():
        raise ValueError("Invalid share multiplier")
    actions["terminal_value"] = pd.to_numeric(actions.terminal_value, errors="raise")
    terminal = actions.terminal_value.dropna()
    if (terminal < 0).any() or not np.isfinite(terminal).all():
        raise ValueError("Invalid terminal proceeds")
    if (actions.payment_date.notna() & (actions.payment_date < actions.date)).any():
        raise ValueError("Payment date precedes ex-date")
    return prices.sort_values(["date", "security_id"]).reset_index(drop=True), actions.sort_values(
        ["date", "security_id"]).reset_index(drop=True)


def coverage_report(prices, actions, calendar, notes=None):
    last = prices.groupby("security_id").date.max()
    unresolved = last[last < calendar.max()].index.tolist()
    terminal = set(actions.loc[actions.terminal_value.notna(), "security_id"])
    gaps = sum(int(((calendar >= frame.date.min()) & (calendar <= frame.date.max())).sum()) - len(frame)
               for _, frame in prices.groupby("security_id"))
    return {
        "rows": len(prices), "securities": int(prices.security_id.nunique()),
        "sessions": len(calendar), "start": str(calendar.min().date()), "end": str(calendar.max().date()),
        "missing_ohlc_rows": int(prices[["open", "high", "low", "close"]].isna().any(axis=1).sum()),
        "unverified_identity_rows": int((~prices.identity_verified).sum()),
        "action_events": len(actions),
        "unknown_dividend_payment_dates": int((actions.cash_dividend.ne(0) & actions.payment_date.isna()).sum()),
        "suspension_or_missing_session_rows": gaps,
        "unresolved_terminal_securities": [sid for sid in unresolved if sid not in terminal],
        "notes": list(notes or []) + [
            "Historical revisions/knowledge timestamps are not supplied by TWSE daily reports.",
            "Borrow availability, recalls, fees and opening-auction fillability are simulated.",
            "Historical commission/tax schedules require independent verification.",
        ],
        "provisional": True,
    }


def validate_data(config):
    root = Path(config["paths"]["raw"])
    out = Path(config["paths"]["normalized"])
    out.mkdir(parents=True, exist_ok=True)
    prices, actions, notes, sources = [], [], [], set()
    for meta in sorted(root.glob("*.meta.json")):
        info, payload = read_response(meta)
        sources.add(info.get("source", "UNKNOWN"))
        endpoint = info["endpoint"]
        if endpoint == "afterTrading/MI_INDEX":
            frame = parse_prices(payload, info["params"]["date"])
            if not frame.empty:
                prices.append(frame)
        elif endpoint == "exRight/TWT49U":
            try:
                actions.append(parse_actions(payload))
            except ValueError as error:
                notes.append(f"{meta.name}: {error}")
    if not prices:
        raise ValueError("No cached prices; fetch probe data or generate the offline demo")
    # Canonical imports supply explicit terms for events unsupported by the public table.
    imported_actions = root / "actions.csv"
    if imported_actions.exists():
        imported = pd.read_csv(imported_actions, dtype={"security_id": str})
        imported["date"] = pd.to_datetime(imported.date)
        imported_keys = pd.MultiIndex.from_frame(imported[["date", "security_id"]])
        actions = [frame.assign(date=pd.to_datetime(frame.date)) for frame in actions if not frame.empty]
        actions = [frame[~pd.MultiIndex.from_frame(frame[["date", "security_id"]]).isin(imported_keys)]
                   for frame in actions]
        actions.append(imported)
    master = root / "securities.csv"
    securities = pd.read_csv(master, dtype={"security_id": str}) if master.exists() else None
    action_frame = pd.concat(actions, ignore_index=True) if actions else None
    price_frame, action_frame = normalize_frames(pd.concat(prices, ignore_index=True), action_frame, securities)
    price_frame = price_frame[price_frame.date.between(config["research"]["start"], config["research"]["end"])]
    calendar_path = root / "calendar.csv"
    if calendar_path.exists():
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar_path)["date"]))
        if calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("Calendar must contain unique sorted sessions")
        calendar = calendar[(calendar >= pd.Timestamp(config["research"]["start"])) &
                            (calendar <= pd.Timestamp(config["research"]["end"]))]
        missing = calendar.difference(price_frame.date.unique())
        if len(missing):
            raise ValueError(f"Missing entire market sessions: {missing.tolist()}")
        if not price_frame.date.isin(calendar).all():
            raise ValueError("Prices fall outside the supplied trading calendar")
    else:
        calendar = pd.DatetimeIndex(sorted(price_frame.date.unique()))
        notes.append("Calendar inferred from observed reports; missing entire market days cannot be detected.")
    if calendar.empty:
        raise ValueError("No sessions in configured research period")
    notes.append("Corporate-action completeness, delisting coverage and identity history are not certified.")
    report = coverage_report(price_frame, action_frame, calendar, notes)
    report["sources"] = sorted(sources)
    price_frame.to_parquet(out / "prices.parquet", index=False)
    action_frame.to_parquet(out / "actions.parquet", index=False)
    pd.DataFrame({"date": calendar}).to_parquet(out / "calendar.parquet", index=False)
    (out / "coverage.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def market_panel(prices, actions, calendar):
    """Keep missing sessions explicit; no forward-filling raw execution prices."""
    index = pd.MultiIndex.from_product([calendar, sorted(prices.security_id.unique())],
                                       names=["date", "security_id"])
    panel = prices.set_index(["date", "security_id"]).reindex(index).reset_index()
    panel = panel.merge(actions, on=["date", "security_id"], how="left", validate="one_to_one")
    for column, default in [("cash_dividend", 0.0), ("share_multiplier", 1.0),
                            ("subscription_ratio", 0.0), ("subscription_price", 0.0)]:
        panel[column] = panel[column].fillna(default)
    # Raw observations remain NaN. Valuation prices alone carry across suspension.
    panel["mark_close"] = np.nan
    for _, frame in panel.groupby("security_id", sort=False):
        last, values, settled = np.nan, [], False
        for row in frame.itertuples():
            if pd.notna(row.terminal_value):
                last = row.terminal_value
                settled = True
            elif settled:
                pass  # Terminal proceeds are held in cash through later label dates.
            elif pd.notna(row.close):
                last = row.close
            elif pd.notna(last):
                last = (last - row.cash_dividend + row.subscription_ratio * row.subscription_price) / row.share_multiplier
            values.append(last)
        panel.loc[frame.index, "mark_close"] = values
    previous = panel.groupby("security_id").mark_close.shift()
    cash = panel.cash_dividend - panel.subscription_ratio * panel.subscription_price
    factor = (panel.mark_close * panel.share_multiplier + cash) / previous
    factor = factor.where(previous.notna(), 1.0)
    factor = factor.mask(previous.eq(0) & panel.mark_close.eq(0), 1.0)
    factor = factor.where(panel.mark_close.notna())
    if (factor.dropna() < 0).any():
        raise ValueError("Negative corporate-action return factor")
    panel["total_close"] = factor.groupby(panel.security_id).cumprod()
    prior_index = panel.groupby("security_id").total_close.shift()
    panel["total_open"] = prior_index * (panel.open * panel.share_multiplier + cash) / previous
    first = previous.isna() & panel.open.notna() & panel.close.notna()
    panel.loc[first, "total_open"] = panel.loc[first, "open"] / panel.loc[first, "close"]
    settled = panel.terminal_value.notna().groupby(panel.security_id).cummax()
    panel.loc[settled, "total_open"] = panel.loc[settled, "total_close"]
    return panel
