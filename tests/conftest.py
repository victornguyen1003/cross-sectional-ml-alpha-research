import json
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest

from alpha_research.data import normalize_frames
from alpha_research.config import load_config
from alpha_research.fetch import store_response
from alpha_research.factors import FX_URL, JKP, RF_URL, fetch_factors


@pytest.fixture
def factor_downloads():
    """Provider-shaped responses used only in temporary offline test caches."""
    dates = pd.bdate_range("2018-01-01", "2022-12-30")
    rng = np.random.default_rng(203)
    def zipped(body):
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("data.csv", body)
        return buffer.getvalue()
    responses = {}
    for name in ["mkt", "market_equity", "be_me", "ret_12_1"]:
        frame = pd.DataFrame({"location": "twn", "name": name, "freq": "daily", "weighting": "vw_cap",
                              "date": dates, "ret": rng.normal(0, 0.01, len(dates))})
        url = f"{JKP}/%5Btwn%5D_%5B{name}%5D_%5Bdaily%5D_%5Bvw_cap%5D.zip"
        responses[url] = zipped(frame.to_csv(index=False))
    responses[FX_URL] = pd.DataFrame({"observation_date": dates, "DEXTAUS": 30.0}).to_csv(index=False).encode()
    rows = [f"{month:%Y%m},0,0,0,0.21" for month in pd.date_range("2018-01-01", "2022-12-01", freq="MS")]
    responses[RF_URL] = zipped("Provider-shaped test fixture\n,Mkt-RF,SMB,HML,RF\n" + "\n".join(rows))
    return responses


@pytest.fixture
def factor_config(tmp_path):
    config = load_config(Path(__file__).parents[1] / "configs/default.toml")
    config["paths"] = {name: str(tmp_path / name) for name in config["paths"]}
    config["research"].update(start="2018-01-01", end="2022-12-30")
    config["fetch"]["delay_seconds"] = 0
    return config


@pytest.fixture
def market():
    dates = pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-06"])
    rows = [{"date": date, "security_id": sid, "open": 100.0, "high": 101.0,
             "low": 99.0, "close": 100.0, "volume": 1000, "value": 100000}
            for date in dates for sid in ["1000", "1001"]]
    prices, actions = normalize_frames(pd.DataFrame(rows))
    targets = pd.DataFrame({"date": [dates[0] - pd.Timedelta(days=1)] * 2,
                            "entry_date": [dates[0]] * 2,
                            "security_id": ["1000", "1001"], "weight": [0.5, -0.5]})
    costs = {"initial_nav": 1000.0, "commission_bps": 0.0, "sell_tax_bps": 0.0,
             "slippage_bps": 0.0, "borrow_rate": 0.0}
    return prices, actions, dates, targets, costs


@pytest.fixture
def cached_research(tmp_path, monkeypatch, factor_downloads):
    """Small test-only cache; accounting events have separate hand-calculated tests."""
    dates = pd.bdate_range("2018-01-01", "2022-12-30")
    identifiers = [str(1000 + i) for i in range(10)]
    rng = np.random.default_rng(103)
    close = 100 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, (len(dates), 10)), axis=0))
    opening = close * np.exp(rng.normal(0, 0.002, close.shape))
    volume = rng.integers(100000, 1000000, close.shape)
    prices = pd.DataFrame({
        "date": np.repeat(dates.strftime("%Y-%m-%d"), 10), "security_id": np.tile(identifiers, len(dates)),
        "open": opening.ravel(), "close": close.ravel(),
        "high": np.maximum(opening, close).ravel() * 1.005,
        "low": np.minimum(opening, close).ravel() * 0.995,
        "volume": volume.ravel(), "value": (volume * close).ravel(),
    })
    raw = tmp_path / "raw"
    store_response(raw, "afterTrading/MI_INDEX", {"date": "20180101", "type": "ALLBUT0999", "response": "json"},
                   json.dumps({"stat": "OK", "prices": prices.to_dict("records")}).encode(), source="SYNTHETIC_TEST")
    store_response(raw, "exRight/TWT49U", {"startDate": "20180101", "endDate": "20180131", "response": "json"},
                   b'{"stat":"OK","actions":[]}', source="SYNTHETIC_TEST")
    pd.DataFrame({"date": dates}).to_csv(raw / "calendar.csv", index=False)
    pd.DataFrame(columns=["date", "security_id", "cash_dividend", "share_multiplier",
                          "subscription_ratio", "subscription_price", "payment_date", "terminal_value"]).to_csv(raw / "actions.csv", index=False)
    pd.DataFrame({"security_id": identifiers, "valid_from": "2018-01-01", "valid_to": "2022-12-31",
                  "security_type": "common"}).to_csv(raw / "securities.csv", index=False)
    content = (Path(__file__).parents[1] / "configs" / "default.toml").read_text()
    replacements = {"../data/raw": "raw", "../data/normalized": "normalized", "../data/dataset": "dataset",
                    "../artifacts/runs": "runs", "../data/factors": "factors",
                    "delay_seconds = 3.0": "delay_seconds = 0.0",
                    "2010-01-01": "2018-01-01", "2025-12-31": "2022-12-30",
                    "2011-01-01": "2019-01-01", "2020-01-01": "2021-01-01", "2024-01-01": "2022-01-01",
                    '["2010-01-04", "2025-10-01"]': '["2018-01-01"]', "cv_splits = 3": "cv_splits = 2",
                    "[0.1, 1.0, 10.0]": "[1.0]", "[0.0001, 0.001, 0.01]": "[0.001]",
                    "[0.25, 0.75]": "[0.5]", "[7, 15]": "[7]", "[0.03, 0.1]": "[0.1]",
                    "[100]": "[10]", "[0.0, 10.0, 25.0]": "[0.0, 10.0]", "[0.0, 0.03, 0.06]": "[0.03]"}
    for before, after in replacements.items():
        content = content.replace(before, after)
    path = tmp_path / "research.toml"
    path.write_text(content)
    with monkeypatch.context() as patch:
        patch.setattr("requests.Session.get", lambda self, url, **kwargs: SimpleNamespace(
            content=factor_downloads[url], raise_for_status=lambda: None))
        fetch_factors(load_config(path))
    return path
