"""Polite TWSE requests with immutable response bytes and verified offline replay."""

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.twse.com.tw/rwd/en/"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def cache_key(endpoint, params):
    request = json.dumps([endpoint, params], sort_keys=True).encode()
    return hashlib.sha256(request).hexdigest()


def store_response(root, endpoint, params, body, source="TWSE"):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    key = cache_key(endpoint, params)
    path = root / f"{key}.json"
    meta = root / f"{key}.meta.json"
    if path.exists() or meta.exists():
        raise FileExistsError(f"Refusing to overwrite cached response {key}")
    # The metadata is the completion marker. An interrupted write is never replayed.
    path.write_bytes(body)
    meta.write_text(json.dumps({
        "endpoint": endpoint, "params": params, "source": source,
        "retrieved_at": datetime.now(timezone.utc).isoformat(),
        "sha256": digest(path),
    }, indent=2) + "\n")
    return json.loads(body)


def read_response(meta):
    meta = Path(meta)
    info = json.loads(meta.read_text())
    path = meta.with_name(meta.name.replace(".meta.json", ".json"))
    if not path.exists() or digest(path) != info["sha256"]:
        raise ValueError(f"Missing or corrupt raw cache: {path}")
    return info, json.loads(path.read_bytes())


def fetch_response(root, endpoint, params, settings, offline=False, session=None):
    root = Path(root)
    meta = root / f"{cache_key(endpoint, params)}.meta.json"
    if meta.exists():
        return read_response(meta)[1]
    if offline:
        raise FileNotFoundError(f"No cached response for {endpoint} {params}")
    path = root / f"{cache_key(endpoint, params)}.json"
    if path.exists():
        raise ValueError(f"Incomplete cache entry; inspect and remove before retry: {path}")
    client = session or requests.Session()
    last = None
    for attempt in range(settings["retries"]):
        time.sleep(settings["delay_seconds"] * (attempt + 1))
        try:
            response = client.get(BASE + endpoint, params=params,
                                  timeout=settings["timeout_seconds"])
            response.raise_for_status()
            payload = response.json()
            status = str(payload.get("stat", ""))
            if status != "OK" and not any(s in status for s in ["沒有", "查無", "No data"]):
                raise ValueError(f"Unexpected TWSE status {status!r}")
            return store_response(root, endpoint, params, response.content)
        except (requests.RequestException, ValueError) as error:
            last = error
    raise RuntimeError(f"TWSE request failed after retries: {endpoint} {params}") from last


def fetch(config, offline=False, bulk=False):
    research, settings = config["research"], config["fetch"]
    if bulk:
        dates = pd.date_range(research["start"], research["end"], freq="B")
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
    return counts
