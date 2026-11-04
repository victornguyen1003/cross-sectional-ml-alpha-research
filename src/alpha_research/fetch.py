"""Polite TWSE requests with immutable response bytes and verified offline replay."""

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.twse.com.tw/rwd/en/"
ISIN = "https://isin.twse.com.tw/isin/class_main.jsp"
LISTINGS = "https://www.twse.com.tw/rwd/zh/company/newlisting"
DELISTINGS = "company/suspendListing"
ACTION_DETAILS = "https://www.twse.com.tw/rwd/zh/exRight/TWT49UDetail"
ACTION_REFERENCES = "https://www.twse.com.tw/rwd/zh/exRight/TWT49U"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(endpoint, params):
    request = json.dumps([endpoint, params], sort_keys=True).encode()
    return hashlib.sha256(request).hexdigest()


def store_response(root, endpoint, params, body, source="TWSE", encoding=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(endpoint, params)
    path = root / f"{key}.{'html' if encoding else 'json'}"
    meta = root / f"{key}.meta.json"
    if path.exists() or meta.exists():
        raise FileExistsError(f"Refusing to overwrite cached response {key}")
    # The metadata is the completion marker. An interrupted write is never replayed.
    path.write_bytes(body)
    meta.write_text(json.dumps({
        "endpoint": endpoint, "params": params, "source": source,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": digest(path),
        **({"encoding": encoding} if encoding else {}),
    }, indent=2) + "\n")
    return body.decode(encoding) if encoding else json.loads(body)


def read_response(meta):
    meta = Path(meta)
    info = json.loads(meta.read_text())
    path = meta.with_name(meta.name.replace(".meta.json", ".html" if info.get("encoding") else ".json"))
    if not path.exists() or digest(path) != info["sha256"]:
        raise ValueError(f"Missing or corrupt raw cache: {path}")
    body = path.read_bytes()
    return info, body.decode(info["encoding"]) if info.get("encoding") else json.loads(body)


def fetch_response(root, endpoint, params, settings, offline=False, session=None, encoding=None):
    root = Path(root)
    meta = root / f"{cache_key(endpoint, params)}.meta.json"
    if meta.exists():
        return read_response(meta)[1]
    if offline:
        raise FileNotFoundError(f"No cached response for {endpoint} {params}")
    path = root / f"{cache_key(endpoint, params)}.{'html' if encoding else 'json'}"
    if path.exists():
        raise ValueError(f"Incomplete cache entry; inspect and remove before retry: {path}")
    client = session or requests.Session()
    last = None
    for attempt in range(settings["retries"]):
        time.sleep(settings["delay_seconds"] * (attempt + 1))
        try:
            response = client.get(endpoint if endpoint.startswith("https://") else BASE + endpoint, params=params,
                                  timeout=settings["timeout_seconds"])
            response.raise_for_status()
            if encoding:
                response.content.decode(encoding)
                return store_response(root, endpoint, params, response.content, encoding=encoding)
            payload = response.json()
            status = str(payload.get("stat", payload.get("status", "")))
            no_data = (status.strip().casefold() in {"no data", "no data!", "no data."} or
                       any(s in status for s in ["沒有", "查無"]))
            if status.upper() != "OK" and not no_data:
                raise ValueError(f"Unexpected TWSE status {status!r}")
            return store_response(root, endpoint, params, response.content)
        except (requests.RequestException, ValueError) as error:
            last = error
    raise RuntimeError(f"TWSE request failed after retries: {endpoint} {params}") from last


def fetch_securities(config, offline=False):
    """Cache official classifications and dated listing/delisting records."""
    root, settings = config["paths"]["raw"], config["fetch"]
    with requests.Session() as session:
        fetch_response(root, ISIN, {}, settings, offline, session, encoding="big5hkscs")
        fetch_response(root, DELISTINGS, {"response": "json"}, settings, offline, session)
        for year in range(2000, pd.Timestamp(config["research"]["end"]).year + 1):
            fetch_response(root, LISTINGS, {"date": str(year), "selectType": "", "response": "json"},
                           settings, offline, session)
    return {"listing_years": pd.Timestamp(config["research"]["end"]).year - 1999}


def fetch_actions(config, offline=False):
    """Use explicit cash amounts, fetching individual terms for rights/mixed events."""
    requests_count = 0
    with requests.Session() as session:
        for month in pd.period_range(config["research"]["start"], config["research"]["end"], freq="M"):
            params = {"startDate": month.start_time.strftime("%Y%m%d"),
                      "endDate": month.end_time.strftime("%Y%m%d"), "response": "json"}
            payload = fetch_response(config["paths"]["raw"], ACTION_REFERENCES, params,
                                     config["fetch"], offline, session)
            fields = payload.get("fields", [])
            if not payload.get("data"):
                continue
            if not {"資料日期", "股票代號", "權/息", "權值+息值"}.issubset(fields):
                raise ValueError("Unrecognized TWSE historical action-reference schema")
            for row in payload["data"]:
                year, month_number, day = map(int, re.findall(r"\d+", row[fields.index("資料日期")]))
                date = pd.Timestamp(year + 1911, month_number, day)
                if date.to_period("M") != month:
                    raise ValueError("Action response contains dates outside its requested month")
                if str(row[fields.index("權/息")]).strip() == "息":
                    continue
                sid = str(row[fields.index("股票代號")]).strip()
                fetch_response(config["paths"]["raw"], ACTION_DETAILS,
                               {"STK_NO": sid, "T1": date.strftime("%Y%m%d"), "response": "json"},
                               config["fetch"], offline, session)
                requests_count += 1
    return {"action_detail_requests": requests_count}


def calendar_dates(payload, month):
    """Read actual sessions, including Saturday make-up trading, from FMTQIK."""
    fields = payload.get("fields", [])
    if "Date" not in fields or not payload.get("data"):
        raise ValueError(f"Missing monthly TWSE trading calendar: {month}")
    dates = pd.DatetimeIndex(pd.to_datetime([row[fields.index("Date")] for row in payload["data"]],
                                          format="%Y/%m/%d"))
    if dates.has_duplicates or not dates.is_monotonic_increasing or not (dates.to_period("M") == month).all():
        raise ValueError(f"Invalid monthly TWSE calendar: {month}")
    return dates


def fetch_calendar(config, offline=False):
    """Cache an independent calendar for every month of the research period."""
    research = config["research"]
    dates = []
    with requests.Session() as session:
        for month in pd.period_range(research["start"], research["end"], freq="M"):
            payload = fetch_response(config["paths"]["raw"], "afterTrading/FMTQIK",
                                     {"date": month.start_time.strftime("%Y%m%d"), "response": "json"},
                                     config["fetch"], offline, session)
            dates.extend(calendar_dates(payload, month))
    dates = pd.DatetimeIndex(dates)
    return dates[(dates >= pd.Timestamp(research["start"])) & (dates <= pd.Timestamp(research["end"]))]


def fetch(config, offline=False, bulk=False):
    research, settings = config["research"], config["fetch"]
    if bulk:
        dates = fetch_calendar(config, offline)
        dates = dates[(dates >= pd.Timestamp(research["start"])) & (dates <= pd.Timestamp(research["end"]))]
    else:
        dates = pd.to_datetime(settings["probe_dates"])
    counts = {"price_requests": 0, "action_requests": 0}
    with requests.Session() as session:
        for date in dates:
            params = {"date": date.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"}
            fetch_response(config["paths"]["raw"], "afterTrading/MI_INDEX", params,
                           settings, offline, session)
            counts["price_requests"] += 1
        # Actions are queried one month at a time to avoid endpoint range limits.
        months = sorted(set(pd.Timestamp(d).to_period("M") for d in dates))
        for month in months:
            params = {"startDate": month.start_time.strftime("%Y%m%d"),
                      "endDate": month.end_time.strftime("%Y%m%d"), "response": "json"}
            fetch_response(config["paths"]["raw"], "exRight/TWT49U", params,
                           settings, offline, session)
            counts["action_requests"] += 1
    if bulk:
        counts.update(fetch_actions(config, offline))
    return counts
