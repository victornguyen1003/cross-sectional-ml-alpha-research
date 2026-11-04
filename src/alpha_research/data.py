"""Normalize observed history without substituting today's security universe."""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import ACTION_DETAILS, ACTION_REFERENCES, DELISTINGS, ISIN, LISTINGS, calendar_dates, read_response

PRICE_COLUMNS = ["date", "security_id", "open", "high", "low", "close", "volume", "value"]
ACTION_COLUMNS = ["date", "security_id", "cash_dividend", "share_multiplier",
                  "subscription_ratio", "subscription_price", "payment_date", "terminal_value"]


def number(value):
    text = re.sub(r"<[^>]*>", "", str(value)).replace(",", "").strip()
    return pd.to_numeric(text, errors="coerce")


def parse_date(value):
    text = str(value).strip()
    if re.fullmatch(r"\d{2,3}[年./]\d{1,2}[月./]\d{1,2}日?", text):
        year, month, day = map(int, re.findall(r"\d+", text))
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


def parse_actions(payload, date=None, security_id=None):
    if "actions" in payload:
        return pd.DataFrame(payload["actions"], columns=ACTION_COLUMNS)
    fields = payload.get("fields", [])
    rows = payload.get("data", [])
    if not rows:
        return pd.DataFrame(columns=ACTION_COLUMNS)
    if "按股東持股比例每千股認購" in fields or "G. 按特別股股東持股比例每千股有償認股" in fields:
        if date is None or security_id is None:
            raise ValueError("Historical action detail requires its requested date and security")
        ordinary = "按股東持股比例每千股認購" in fields
        stock_field = "A. 按普通股股東持股比例每千股無償配股" if ordinary else "F. 按特別股股東持股比例每千股無償配股"
        subscription_field = "按股東持股比例每千股認購" if ordinary else "G. 按特別股股東持股比例每千股有償認股"
        required = {"股票代號", "(每股配發現金股利)除息", stock_field, subscription_field, "每股認購金額"}
        if not required.issubset(fields):
            raise ValueError(f"Unrecognized historical action-detail fields: {fields}")
        row = dict(zip(fields, rows[0]))
        if len(rows) != 1 or str(row["股票代號"]).strip() != security_id:
            raise ValueError("Historical action detail does not match the requested security")
        def term(name, unit):
            value = str(row[name]).strip().replace(",", "")
            match = re.fullmatch(r"([0-9]+(?:\.[0-9]+)?)\s*" + unit, value)
            if not match:
                raise ValueError(f"Missing or unsupported action term {name}: {value}")
            return float(match[1])
        subscription = term(subscription_field, "股") / 1000
        stock = term(stock_field, "股") / 1000
        return pd.DataFrame([{"date": parse_date(date), "security_id": security_id,
                              "cash_dividend": term("(每股配發現金股利)除息", "元[／/]股"),
                              "share_multiplier": 1 + stock + subscription,
                              "subscription_ratio": subscription,
                              "subscription_price": term("每股認購金額", "元[／/]股"),
                              "payment_date": pd.NaT, "terminal_value": np.nan}], columns=ACTION_COLUMNS)
    if "Ex-right/Ex-dividend Reference Price" in fields or "除權息參考價" in fields:
        if {"資料日期", "股票代號", "權/息", "權值+息值"}.issubset(fields):
            # For an explicit cash-only event, TWSE's rights+dividend amount is
            # the cash dividend itself. No price difference or rounded quote is used.
            result = []
            for row in rows:
                if str(row[fields.index("權/息")]).strip() == "息":
                    cash = number(row[fields.index("權值+息值")])
                    if not np.isfinite(cash):
                        raise ValueError("Missing explicit cash-dividend amount")
                    result.append({"date": parse_date(row[fields.index("資料日期")]),
                                   "security_id": str(row[fields.index("股票代號")]).strip(),
                                   "cash_dividend": cash, "share_multiplier": 1.0,
                                   "subscription_ratio": 0.0, "subscription_price": 0.0,
                                   "payment_date": pd.NaT, "terminal_value": np.nan})
            return pd.DataFrame(result, columns=ACTION_COLUMNS)
        raise ValueError("Reference-price event has no complete action terms; fetch the historical TWT49UDetail record")
    # Positional guessing would silently corrupt returns when TWSE changes schema.
    def field(*names):
        for name in names:
            if name in fields:
                return fields.index(name)
        raise ValueError(f"Unrecognized corporate-action fields: {fields}")
    date_i = field("Date", "Ex-right Date", "Ex-right/Ex-dividend Date", "除權息日期")
    id_i = field("Code", "Security Code", "Stock Code", "股票代號")
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
    unresolved = last[last < prices.date.max()].index.tolist()
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
    from .universe import security_history

    root, out = Path(config["paths"]["raw"]), Path(config["paths"]["normalized"])
    prices, actions, notes, sources = [], [], [], set()
    references, calendars, listings, schema_errors, detail_errors, details = set(), {}, [], [], [], []
    action_months = set()
    unavailable_details = set()
    isin_html, delistings = None, {"data": []}
    for meta in sorted(root.glob("*.meta.json")):
        info, payload = read_response(meta)
        sources.add(info.get("source", "UNKNOWN"))
        endpoint, params = info["endpoint"], info["params"]
        if endpoint == "afterTrading/MI_INDEX":
            if payload.get("date") and payload["date"] != params["date"]:
                raise ValueError(f"Price response date does not match request: {meta.name}")
            frame = parse_prices(payload, params["date"])
            if not frame.empty:
                prices.append(frame)
        elif endpoint == "afterTrading/FMTQIK":
            month = pd.Timestamp(params["date"]).to_period("M")
            calendars[month] = calendar_dates(payload, month)
        elif endpoint == ISIN:
            isin_html = payload
        elif endpoint == LISTINGS:
            listings.append(payload)
        elif endpoint == DELISTINGS:
            delistings = payload
        elif endpoint == ACTION_DETAILS:
            try:
                detail = parse_actions(payload, params["T1"], params["STK_NO"])
                if detail.empty:
                    unavailable_details.add((parse_date(params["T1"]), params["STK_NO"]))
                else:
                    details.append(detail)
            except ValueError as error:
                detail_errors.append({"date": params["T1"], "security_id": params["STK_NO"], "error": str(error)})
        elif endpoint in {"exRight/TWT49U", ACTION_REFERENCES}:
            action_months.update(pd.period_range(parse_date(params["startDate"]), parse_date(params["endDate"]), freq="M"))
            if "actions" in payload:
                actions.append(parse_actions(payload))
            elif payload.get("data"):
                fields = payload.get("fields", [])
                if {"Date", "Security Code", "Ex-right/Ex-dividend Reference Price"}.issubset(fields):
                    references.update((parse_date(row[fields.index("Date")]), str(row[fields.index("Security Code")]).strip())
                                      for row in payload["data"])
                elif {"資料日期", "股票代號", "除權息參考價"}.issubset(fields):
                    references.update((parse_date(row[fields.index("資料日期")]), str(row[fields.index("股票代號")]).strip())
                                      for row in payload["data"])
                    actions.append(parse_actions(payload))
                else:
                    schema_errors.append({"file": meta.name, "fields": fields})
    if not prices:
        raise ValueError("No cached prices; fetch TWSE data first")
    raw_prices = pd.concat(prices, ignore_index=True)
    period = config["research"]
    raw_prices["date"] = pd.to_datetime(raw_prices.date)
    raw_prices = raw_prices[pd.to_datetime(raw_prices.date).between(period["start"], period["end"])]
    master = root / "securities.csv"
    securities = (pd.read_csv(master, dtype={"security_id": str}) if master.exists() else
                  security_history(isin_html, listings, delistings, period["end"], raw_prices))
    if details:
        detail_frame = pd.concat(details, ignore_index=True)
        keys = pd.MultiIndex.from_frame(detail_frame[["date", "security_id"]])
        actions = [frame[~pd.MultiIndex.from_frame(frame[["date", "security_id"]]).isin(keys)]
                   for frame in actions if not frame.empty]
        actions.append(detail_frame)
    imported_actions = root / "actions.csv"
    if imported_actions.exists():
        imported = pd.read_csv(imported_actions, dtype={"security_id": str})
        imported["date"] = pd.to_datetime(imported.date)
        imported_keys = pd.MultiIndex.from_frame(imported[["date", "security_id"]])
        actions = [frame[~pd.MultiIndex.from_frame(frame[["date", "security_id"]]).isin(imported_keys)]
                   for frame in actions if not frame.empty]
        actions.append(imported)
    action_frame = pd.concat(actions, ignore_index=True) if actions else None
    price_frame, action_frame = normalize_frames(raw_prices, action_frame, securities)
    action_frame = action_frame[action_frame.date.between(period["start"], period["end"])]
    observed = pd.DatetimeIndex(sorted(price_frame.date.unique()))
    calendar_path = root / "calendar.csv"
    months = pd.period_range(period["start"], period["end"], freq="M")
    missing_months = [str(month) for month in months if month not in calendars]
    missing_action_months = [str(month) for month in months if month not in action_months]
    if calendar_path.exists():
        calendar = pd.DatetimeIndex(pd.to_datetime(pd.read_csv(calendar_path)["date"]))
        if calendar.has_duplicates or not calendar.is_monotonic_increasing:
            raise ValueError("Calendar must contain unique sorted sessions")
        calendar_source, missing_months = "audited calendar.csv", []
    elif calendars:
        calendar = pd.DatetimeIndex(sorted(set().union(*(set(dates) for dates in calendars.values()))))
        # Retain observed dates while the independent monthly calendar is incomplete.
        if missing_months:
            calendar = calendar.union(observed).sort_values()
        calendar_source = "TWSE FMTQIK actual monthly sessions"
    else:
        calendar, calendar_source = observed, "unverified observed-price dates"
    calendar = calendar[(calendar >= pd.Timestamp(period["start"])) & (calendar <= pd.Timestamp(period["end"]))]
    if calendar.empty:
        raise ValueError("No sessions in configured research period")
    if not observed.isin(calendar).all():
        raise ValueError("Prices fall outside the supplied trading calendar")
    missing_sessions = calendar.difference(observed)
    known = set(zip(action_frame.date, action_frame.security_id))
    references = {(date, sid) for date, sid in references if pd.Timestamp(period["start"]) <= date <= pd.Timestamp(period["end"])}
    unresolved = pd.DataFrame(sorted(references - known), columns=["date", "security_id"])
    failed_details = {(parse_date(item["date"]), item["security_id"]) for item in detail_errors}
    unresolved["reason"] = ["unusable downloaded detail" if key in failed_details else
                            "TWSE detail returned no data" if key in unavailable_details else "explicit terms not downloaded"
                            for key in zip(unresolved.date, unresolved.security_id)]
    unresolved["common"] = False
    for row in securities[securities.security_type.eq("common")].itertuples():
        mask = unresolved.security_id.eq(row.security_id) & unresolved.date.between(pd.Timestamp(row.valid_from), pd.Timestamp(row.valid_to))
        unresolved.loc[mask, "common"] = True
    candidates = (price_frame.security_type.eq("common_candidate") & ~price_frame.identity_verified &
                  price_frame.close.notna() & price_frame.volume.gt(0))
    notes.extend([
        "Current ISIN classifications are combined with dated listing/delisting records; historical type continuity and ticker reuse are not fully certified.",
        "Reference-price events are counted separately from usable actions; missing terms are never replaced with zero.",
        "Dividend payment dates, capital reductions, splits and terminal settlement values need additional audited evidence.",
    ])
    if missing_months or len(missing_sessions):
        notes.append("Historical collection is incomplete; dataset construction is blocked until full independent calendar coverage is present.")
        notes.append("Last-quote terminal diagnostics are provisional: missing quotes can reflect incomplete collection or suspension, not a confirmed delisting.")
    action_reference_complete = not missing_action_months or imported_actions.exists()
    if not action_reference_complete:
        notes.append("Monthly corporate-action reference coverage is incomplete; missing reports cannot establish absence of events.")
    report = coverage_report(price_frame, action_frame, calendar, notes)
    report.update({
        "sources": sorted(sources), "calendar_source": calendar_source,
        "calendar_complete": not missing_months, "missing_calendar_months": missing_months,
        "expected_sessions": len(calendar) if not missing_months else None,
        "sessions": len(observed), "missing_price_sessions": missing_sessions.strftime("%Y-%m-%d").tolist(),
        "observed_start": str(observed.min().date()), "observed_end": str(observed.max().date()),
        "collection_complete": not missing_months and not len(missing_sessions),
        "security_master": "audited securities.csv" if master.exists() else "TWSE listing/delisting history and ISIN classifications",
        "verified_identity_rows": int(price_frame.identity_verified.sum()),
        "unverified_common_candidate_rows": int(candidates.sum()),
        "unverified_common_candidate_securities": sorted(price_frame.loc[candidates, "security_id"].unique().tolist()),
        "reference_action_events": len(references), "unresolved_action_events": len(unresolved),
        "action_reference_complete": action_reference_complete, "missing_action_months": missing_action_months,
        "action_source": "audited actions.csv plus public records" if imported_actions.exists() else "TWSE historical references and details",
        "unresolved_common_action_events": int(unresolved.common.sum()),
        "unresolved_action_reasons": unresolved.reason.value_counts().to_dict(),
        "action_schema_errors": schema_errors, "action_detail_errors": detail_errors,
    })
    report["research_ready"] = bool(report["collection_complete"] and action_reference_complete and not candidates.any() and
                                    not unresolved.common.any() and not schema_errors and not detail_errors)
    out.mkdir(parents=True, exist_ok=True)
    price_frame.to_parquet(out / "prices.parquet", index=False)
    action_frame.to_parquet(out / "actions.parquet", index=False)
    securities.to_parquet(out / "securities.parquet", index=False)
    unresolved.to_parquet(out / "unresolved_actions.parquet", index=False)
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


def build_dataset(config):
    """Build point-in-time samples and record their normalized input provenance."""
    from .features import FEATURES, build_features
    from .labels import build_labels
    from .provenance import code_fingerprint, config_fingerprint, file_checksums, write_json
    from .universe import select_universe

    root, out = Path(config["paths"]["normalized"]), Path(config["paths"]["dataset"])
    coverage = json.loads((root / "coverage.json").read_text())
    if not coverage.get("research_ready", False):
        raise ValueError("Data quality is unresolved; complete calendar, price, identity and action checks before building the dataset")
    out.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(root / "prices.parquet")
    actions = pd.read_parquet(root / "actions.parquet")
    calendar = pd.DatetimeIndex(pd.read_parquet(root / "calendar.parquet").date)
    panel = market_panel(prices, actions, calendar)
    panel = select_universe(panel, config["research"])
    panel = build_features(panel)
    samples = build_labels(panel, calendar, config["research"])
    if samples.empty:
        raise ValueError("No eligible samples; inspect history, security types and liquidity")
    samples.to_parquet(out / "samples.parquet", index=False)
    panel[["date", "security_id", "security_type", "identity_verified", "history_sessions",
           "liquidity", "eligible", "selected"]].to_parquet(out / "eligibility.parquet", index=False)
    summary = {"rows": len(samples), "signal_dates": samples.date.nunique(),
               "missing_labels": int((~samples.label_available).sum()), "features": FEATURES,
               "config_fingerprint": config_fingerprint(config),
               "code_fingerprint": code_fingerprint(),
               "normalized_checksums": file_checksums(root)}
    write_json(out / "dataset.json", summary)
    return summary
