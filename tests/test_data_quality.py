import json
from pathlib import Path

import pandas as pd
import pytest

from alpha_research.data import build_dataset, normalize_frames, parse_actions, validate_data
from alpha_research.config import load_config
from alpha_research.fetch import ACTION_DETAILS, calendar_dates, fetch, store_response
from alpha_research.universe import security_history, select_universe


DETAIL_FIELDS = ["股票代號", "股票名稱", "(每股配發現金股利)除息", "(增資配股) 除權",
                 "A. 按普通股股東持股比例每千股無償配股", "B. 員工紅利轉增資", "C. (有償) 現金增資",
                 "每股認購金額", "a. 公開承銷", "b. 員工認購", " c. 原股東認購", "按股東持股比例每千股認購"]


def test_explicit_historical_subscription_terms_preserve_units():
    payload = {"stat": "ok", "fields": DETAIL_FIELDS, "data": [["3058  ", "立德", "0 元／股", "",
               "50 股", "0 股", "10,000,000 股", "21 元／股", "1,000,000 股", "1,500,000 股", "7,500,000 股", "66.05047560 股"]]}
    action = parse_actions(payload, "20100114", "3058").iloc[0]
    assert action.subscription_ratio == pytest.approx(0.06605047560)
    assert action.share_multiplier == pytest.approx(1.11605047560)
    assert action.subscription_price == 21
    assert action.cash_dividend == 0
    assert pd.isna(action.payment_date)
    with pytest.raises(ValueError, match="requested security"):
        parse_actions(payload, "20100114", "2330")


def test_cash_amount_uses_explicit_dividend_field_not_rounded_reference_prices():
    payload = {"fields": ["資料日期", "股票代號", "除權息參考價", "權/息", "權值+息值"],
               "data": [["114年06月12日", "2330", "145.50", "息", "4.500020"],
                        ["114年06月12日", "3058", "28.26", "權", "0.639420"]]}
    actions = parse_actions(payload)
    assert len(actions) == 1
    assert actions.iloc[0].cash_dividend == 4.500020
    assert actions.iloc[0].date == pd.Timestamp("2025-06-12")
    payload["data"][0][-1] = "--"
    with pytest.raises(ValueError, match="Missing explicit"):
        parse_actions(payload)


def test_preferred_share_detail_has_explicit_separate_distribution_fields():
    payload = {"fields": ["股票代號", "股票名稱", "(每股配發現金股利)除息", "(增資配股) 除權",
                           "F. 按特別股股東持股比例每千股無償配股", "G. 按特別股股東持股比例每千股有償認股", "每股認購金額"],
               "data": [["2002A ", "中鋼特", "1.07 元／股", "", "33 股", "0 股", "0 元／股"]]}
    action = parse_actions(payload, "20100726", "2002A").iloc[0]
    assert action.cash_dividend == 1.07
    assert action.share_multiplier == 1.033
    assert action.subscription_ratio == 0


def test_bulk_uses_exchange_sessions_including_makeup_saturday(monkeypatch, tmp_path):
    calls = []
    def response(root, endpoint, params, *args, **kwargs):
        calls.append((endpoint, params))
        if endpoint == "afterTrading/FMTQIK":
            return {"fields": ["Date"], "data": [["2010/02/06"], ["2010/02/08"]]}
        return {"stat": "OK", "data": []}
    monkeypatch.setattr("alpha_research.fetch.fetch_response", response)
    config = {"paths": {"raw": str(tmp_path)}, "fetch": {},
              "research": {"start": "2010-02-01", "end": "2010-02-28"}}
    fetch(config, bulk=True)
    assert [params["date"] for endpoint, params in calls if endpoint == "afterTrading/MI_INDEX"] == ["20100206", "20100208"]
    with pytest.raises(ValueError, match="Invalid monthly"):
        calendar_dates({"fields": ["Date"], "data": [["2025/01/02"]]}, pd.Period("2010-02", freq="M"))


def test_history_retains_delisted_companies_and_bounds_membership():
    listings = [{"fields": ["公司代號", "股票上市買賣日期"], "data": [["1000", "97.01.02"]]}]
    html = "<table><tr>" + "".join(f"<td>{value}</td>" for value in
            ["1", "TW0001001000", "1001", "Company", "上市", "股票", "", "2012/01/02", "ESVUFR", ""]) + "</tr></table>"
    delisted = {"fields": ["Delisting Date", "Company", "Code"], "data": [["2011/06/01", "Old company", "1000"]]}
    history = security_history(html, listings, delisted, "2025-12-31").set_index("security_id")
    assert history.loc["1000", "valid_from"] == pd.Timestamp("2008-01-02")
    assert history.loc["1000", "valid_to"] == pd.Timestamp("2011-05-31")
    assert history.loc["1001", "valid_from"] == pd.Timestamp("2012-01-02")
    prices = pd.DataFrame({"date": pd.to_datetime(["2011-05-31", "2011-06-01", "2011-06-01"]),
                           "security_id": ["1000", "1000", "1001"], "open": 1, "high": 1,
                           "low": 1, "close": 1, "volume": 100, "value": 100})
    prices, _ = normalize_frames(prices, securities=history.reset_index())
    selected = select_universe(prices, {"min_history": 1, "liquidity_window": 1, "universe_size": 500})
    assert selected.selected.tolist() == [True, False, False]


def test_retired_company_records_and_reused_codes_do_not_require_survival():
    observations = pd.DataFrame({"security_id": ["2311", "2301", "2301", "9188"],
                                 "date": pd.to_datetime(["2010-01-04", "2010-01-04", "2025-12-31", "2010-01-04"])})
    html = "<table><tr>" + "".join(f"<td>{value}</td>" for value in
            ["1", "TW0002301009", "2301", "Company", "上市", "股票", "", "1995/11/17", "ESVUFR", ""]) + "</tr></table>"
    delisted = {"fields": ["Delisting Date", "Company", "Code"],
                "data": [["2018/04/30", "Retired company", "2311"],
                         ["2002/11/04", "Earlier company", "2301"],
                         ["2022/03/18", "Depositary receipt", "9188"]]}
    history = security_history(html, [], delisted, "2025-12-31", observations).set_index("security_id")
    assert history.loc["2311", "security_type"] == "common"
    assert history.loc["2311", "valid_from"] == pd.Timestamp("2010-01-04")
    assert history.loc["2311", "valid_to"] == pd.Timestamp("2018-04-29")
    assert "IPO date unknown" in history.loc["2311", "source"]
    assert history.loc["9188", "security_type"] == "depositary_receipt"
    assert history.loc["2301", "valid_from"] <= observations.date.min()
    assert history.loc["2301", "valid_to"] == pd.Timestamp("2025-12-31")


def test_missing_action_months_are_not_treated_as_no_events(cached_research):
    config = load_config(cached_research)
    (Path(config["paths"]["raw"]) / "actions.csv").unlink()
    report = validate_data(config)
    assert report["collection_complete"]
    assert report["unresolved_action_events"] == 0
    assert "2020-06" in report["missing_action_months"]
    assert not report["action_reference_complete"]
    assert not report["research_ready"]


def test_missing_whole_session_is_reported_and_blocks_dataset(cached_research):
    config = load_config(cached_research)
    path = Path(config["paths"]["raw"]) / "calendar.csv"
    dates = pd.to_datetime(pd.read_csv(path).date).tolist() + [pd.Timestamp("2018-01-06")]
    pd.DataFrame({"date": sorted(dates)}).to_csv(path, index=False)
    report = validate_data(config)
    assert report["calendar_complete"]
    assert report["missing_price_sessions"] == ["2018-01-06"]
    assert not report["research_ready"]
    with pytest.raises(ValueError, match="Data quality is unresolved"):
        build_dataset(config)


def test_known_reference_events_are_not_confused_with_parser_errors(cached_research):
    config = load_config(cached_research)
    body = {"fields": ["Date", "Security Code", "Ex-right/Ex-dividend Reference Price"],
            "data": [["2021/01/04", "1000", "90.00"]]}
    store_response(config["paths"]["raw"], "exRight/TWT49U",
                   {"startDate": "20210101", "endDate": "20210131", "response": "json"},
                   json.dumps(body).encode(), source="SYNTHETIC_TEST")
    report = validate_data(config)
    assert report["reference_action_events"] == report["unresolved_common_action_events"] == 1
    assert report["action_schema_errors"] == []
    assert not report["research_ready"]
    detail = {"fields": DETAIL_FIELDS, "data": [["1000", "Test", "2 元／股", "", "0 股", "0 股", "0 股",
               "0 元／股", "0 股", "0 股", "0 股", "0 股"]]}
    store_response(config["paths"]["raw"], ACTION_DETAILS,
                   {"STK_NO": "1000", "T1": "20210104", "response": "json"},
                   json.dumps(detail).encode(), source="SYNTHETIC_TEST")
    report = validate_data(config)
    assert report["action_events"] == 1
    assert report["unresolved_common_action_events"] == 0
    assert report["research_ready"]


def test_unavailable_historical_detail_is_distinct_from_parser_failure(cached_research):
    config = load_config(cached_research)
    body = {"fields": ["Date", "Security Code", "Ex-right/Ex-dividend Reference Price"],
            "data": [["2021/01/04", "1000", "90.00"]]}
    store_response(config["paths"]["raw"], "exRight/TWT49U",
                   {"startDate": "20210101", "endDate": "20210131", "response": "json"},
                   json.dumps(body).encode(), source="SYNTHETIC_TEST")
    store_response(config["paths"]["raw"], ACTION_DETAILS,
                   {"STK_NO": "1000", "T1": "20210104", "response": "json"},
                   b'{"stat":"No Data!"}', source="SYNTHETIC_TEST")
    report = validate_data(config)
    assert report["unresolved_action_reasons"] == {"TWSE detail returned no data": 1}
    assert report["action_detail_errors"] == []
    assert not report["research_ready"]
