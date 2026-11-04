import json

import numpy as np
import pandas as pd
import pytest

from alpha_research.data import market_panel, normalize_frames, number, parse_actions, parse_date, parse_prices
from alpha_research.fetch import fetch_response, read_response, store_response


def test_official_price_schema_and_units():
    fields = ["Security Code", "Trade Volume", "Trade Value", "Opening Price",
              "Highest Price", "Lowest Price", "Closing Price"]
    frame = parse_prices({"stat": "OK", "tables": [{"fields": fields, "data": [
        ["2330", "1,000", "100,000", "100", "105", "99", "102"]]}]}, "20251001")
    assert frame.iloc[0].volume == 1000  # Shares, not lots of 1,000 shares.
    assert frame.iloc[0].value == 100000
    assert frame.iloc[0].security_id == "2330"
    assert parse_date("114/10/01") == pd.Timestamp("2025-10-01")
    assert np.isnan(number("--"))
    with pytest.raises(ValueError, match="no recognized"):
        parse_prices({"stat": "OK", "tables": []}, "20251001")


@pytest.mark.parametrize("status", ["No Data!", "No data", "no data.", "很抱歉，沒有符合條件的資料!"])
def test_holiday_no_data_responses_are_cached(tmp_path, status):
    class Response:
        content = json.dumps({"stat": status}).encode()
        def raise_for_status(self):
            pass
        def json(self):
            return json.loads(self.content)
    class Session:
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            return Response()
    session = Session()
    settings = {"retries": 3, "delay_seconds": 0, "timeout_seconds": 1}
    params = {"date": "20100101", "response": "json"}
    payload = fetch_response(tmp_path, "afterTrading/MI_INDEX", params, settings, session=session)
    assert payload["stat"] == status
    assert parse_prices(payload, "20100101").empty
    assert session.calls == 1
    assert fetch_response(tmp_path, "afterTrading/MI_INDEX", params, settings, offline=True) == payload


def test_reference_prices_are_not_guessed_as_total_return_terms():
    with pytest.raises(ValueError, match="corporate-action fields"):
        parse_actions({"fields": ["Date", "Security Code", "Opening Reference Price"],
                       "data": [["2025/10/02", "2330", "100"]]})


def test_cache_offline_resume_integrity_and_retries(tmp_path):
    params = {"date": "20200102"}
    body = b'{"stat":"OK","prices":[]}'
    store_response(tmp_path, "test", params, body)
    settings = {"retries": 2, "delay_seconds": 0, "timeout_seconds": 1}
    assert fetch_response(tmp_path, "test", params, settings, offline=True)["stat"] == "OK"
    with pytest.raises(FileNotFoundError):
        fetch_response(tmp_path, "test", {"date": "missing"}, settings, offline=True)
    meta = next(tmp_path.glob("*.meta.json"))
    info, _ = read_response(meta)
    raw = meta.with_name(meta.name.replace(".meta.json", ".json"))
    raw.write_text("{}")
    with pytest.raises(ValueError, match="corrupt"):
        read_response(meta)
    assert info["params"] == params
    class Response:
        content = body
        def raise_for_status(self):
            pass
        def json(self):
            return json.loads(self.content)
    class Session:
        calls = 0
        def get(self, *args, **kwargs):
            self.calls += 1
            if self.calls == 1:
                raise ValueError("temporary invalid response")
            return Response()
    session = Session()
    fetch_response(tmp_path, "retry", params, settings, session=session)
    fetch_response(tmp_path, "retry", params, settings, session=session)
    assert session.calls == 2


def test_duplicate_invalid_prices_and_historical_identity(market):
    prices, _, dates, _, _ = market
    with pytest.raises(ValueError, match="Duplicate"):
        normalize_frames(pd.concat([prices, prices]))
    bad = prices.copy()
    bad.loc[0, "open"] = -1
    with pytest.raises(ValueError, match="Non-positive"):
        normalize_frames(bad)
    master = pd.DataFrame([{"security_id": "1000", "valid_from": dates[1],
                            "valid_to": dates[-1], "security_type": "common"}])
    frame, _ = normalize_frames(prices, securities=master)
    assert not frame.iloc[0].identity_verified
    assert frame.loc[(frame.security_id == "1000") & (frame.date == dates[-1]), "identity_verified"].all()


def test_dividend_and_split_preserve_economic_total_return(market):
    prices, _, dates, _, _ = market
    prices.loc[(prices.security_id == "1000") & (prices.date >= dates[1]), ["open", "close", "high", "low"]] = [90, 90, 91, 89]
    prices.loc[(prices.security_id == "1001") & (prices.date >= dates[1]), ["open", "close", "high", "low"]] = [50, 50, 51, 49]
    actions = pd.DataFrame([
        {"date": dates[1], "security_id": "1000", "cash_dividend": 10, "share_multiplier": 1},
        {"date": dates[1], "security_id": "1001", "cash_dividend": 0, "share_multiplier": 2},
    ])
    prices, actions = normalize_frames(prices, actions)
    panel = market_panel(prices, actions, dates)
    assert np.allclose(panel.total_close, 1)
    assert np.allclose(panel.total_open, 1)


def test_suspended_ex_date_and_zero_terminal_recovery(market):
    prices, _, dates, _, _ = market
    prices = prices[~((prices.security_id == "1000") & (prices.date >= dates[1]))]
    actions = pd.DataFrame([{"date": dates[1], "security_id": "1000", "cash_dividend": 0,
                             "share_multiplier": 2}])
    prices, actions = normalize_frames(prices, actions)
    panel = market_panel(prices, actions, dates)
    suspended = panel[panel.security_id == "1000"]
    assert suspended.mark_close.tolist() == [100, 50, 50]
    assert suspended.total_close.tolist() == pytest.approx([1, 1, 1])
    actions["terminal_value"] = 0.0
    panel = market_panel(prices, actions, dates)
    terminal = panel[panel.security_id == "1000"]
    assert terminal.total_close.tolist() == pytest.approx([1, 0, 0])
    assert terminal.total_open.iloc[-1] == 0
