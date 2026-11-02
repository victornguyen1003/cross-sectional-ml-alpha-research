"""Generate synthetic cached responses; no synthetic values are real TWSE data."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from alpha_research.fetch import store_response


def generate(output):
    output = Path(output).resolve()
    raw = output / "raw"
    if raw.exists():
        raise FileExistsError(f"Demo already exists: {raw}; choose a new output directory")
    raw.mkdir(parents=True)
    dates = pd.bdate_range("2018-01-01", "2022-12-30")
    rng = np.random.default_rng(103)
    sids = [str(1000 + i) for i in range(20)]
    previous = np.linspace(30, 150, len(sids))
    action_rows = [
        {"date": "2020-06-01", "security_id": "1000", "cash_dividend": 0.5,
         "share_multiplier": 1.0, "subscription_ratio": 0.0, "subscription_price": 0.0,
         "payment_date": "2020-06-15", "terminal_value": None},
        {"date": "2021-03-01", "security_id": "1002", "cash_dividend": 0.0,
         "share_multiplier": 2.0, "subscription_ratio": 0.0, "subscription_price": 0.0,
         "payment_date": None, "terminal_value": None},
        {"date": "2021-06-01", "security_id": "1001", "cash_dividend": 0.0,
         "share_multiplier": 1.0, "subscription_ratio": 0.0, "subscription_price": 0.0,
         "payment_date": None, "terminal_value": 35.0},
    ]
    for date in dates:
        date_text = date.strftime("%Y-%m-%d")
        for action in action_rows:
            if action["date"] == date_text:
                i = sids.index(action["security_id"])
                previous[i] = (previous[i] - action["cash_dividend"]) / action["share_multiplier"]
        common = rng.normal(0.0001, 0.006)
        opening = previous * np.exp(rng.normal(0, 0.002, len(sids)))
        closing = opening * np.exp(common + rng.normal(0.0001, 0.008, len(sids)))
        volume = rng.integers(100000, 2000000, len(sids))
        rows = []
        for i, sid in enumerate(sids):
            if sid == "1001" and date >= pd.Timestamp("2021-06-01"):
                continue
            if sid == "1003" and pd.Timestamp("2021-08-02") <= date <= pd.Timestamp("2021-08-06"):
                continue
            rows.append({"date": date_text, "security_id": sid,
                         "open": float(opening[i]), "high": float(max(opening[i], closing[i]) * 1.005),
                         "low": float(min(opening[i], closing[i]) * 0.995), "close": float(closing[i]),
                         "volume": int(volume[i]), "value": float(volume[i] * (opening[i] + closing[i]) / 2),
                         "buy_blocked": False, "sell_blocked": False})
        params = {"date": date.strftime("%Y%m%d"), "type": "ALLBUT0999", "response": "json"}
        store_response(raw, "afterTrading/MI_INDEX", params,
                       json.dumps({"stat": "OK", "prices": rows}).encode(), source="SYNTHETIC_DEMO")
        previous = closing
    for month in sorted(set(dates.to_period("M"))):
        params = {"startDate": month.start_time.strftime("%Y%m%d"),
                  "endDate": month.end_time.strftime("%Y%m%d"), "response": "json"}
        rows = [row for row in action_rows if pd.Timestamp(row["date"]).to_period("M") == month]
        store_response(raw, "exRight/TWT49U", params,
                       json.dumps({"stat": "OK", "actions": rows}).encode(), source="SYNTHETIC_DEMO")
    pd.DataFrame({"date": dates}).to_csv(raw / "calendar.csv", index=False)
    pd.DataFrame([{"security_id": sid, "valid_from": "2018-01-01",
                   "valid_to": "2021-05-31" if sid == "1001" else "2022-12-31",
                   "security_type": "common"} for sid in sids]).to_csv(raw / "securities.csv", index=False)
    template = Path(__file__).resolve().parents[1] / "configs" / "default.toml"
    content = template.read_text().replace("../data/raw", "raw").replace("../data/normalized", "normalized")
    content = content.replace("../data/dataset", "dataset").replace("../artifacts/runs", "runs")
    replacements = {"2010-01-01": "2018-01-01", "2025-12-31": "2022-12-30",
                    "2011-01-01": "2019-01-01", "2020-01-01": "2021-01-01",
                    "2024-01-01": "2022-01-01",
                    '["2010-01-04", "2025-10-01"]': '["2018-01-01", "2022-12-30"]'}
    for before, after in replacements.items():
        content = content.replace(before, after)
    (output / "demo.toml").write_text(content)
    return output / "demo.toml"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("artifacts/demo"))
    print(generate(parser.parse_args().output))
