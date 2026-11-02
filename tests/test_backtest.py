import pandas as pd
import pytest

from alpha_research.backtest import simulate
from alpha_research.data import normalize_frames


def test_long_short_pnl_drift_and_actual_notional_costs(market):
    prices, actions, dates, targets, costs = market
    prices.loc[prices.security_id == "1000", "close"] = 110
    prices.loc[prices.security_id == "1001", "close"] = 90
    costs.update(commission_bps=10, sell_tax_bps=30, slippage_bps=20)
    result = simulate(prices, actions, dates[:1], targets, costs)
    daily = result["daily"].iloc[0]
    assert daily.nav == pytest.approx(1095.5)
    assert daily.gross_nav == pytest.approx(1100)
    assert daily.commission == pytest.approx(1)
    assert daily.sell_tax == pytest.approx(1.5)
    assert daily.slippage == pytest.approx(2)
    assert daily.long_pnl == pytest.approx(50)
    assert daily.short_pnl == pytest.approx(50)
    assert daily.net_exposure > 0


def test_short_borrow_uses_calendar_days_and_previous_short_market_value(market):
    prices, actions, dates, targets, costs = market
    costs["borrow_rate"] = 0.365
    result = simulate(prices, actions, dates, targets, costs)
    assert result["daily"].borrow.tolist() == pytest.approx([0, 0.5, 1.5])
    assert result["daily"].nav.iloc[-1] == pytest.approx(998)


@pytest.mark.parametrize("sid", ["1000", "1001"])
def test_dividend_receivable_and_short_manufactured_payment(market, sid):
    prices, _, dates, targets, costs = market
    prices.loc[(prices.security_id == sid) & (prices.date >= dates[1]), ["open", "close", "high", "low"]] = [90, 90, 91, 89]
    actions = pd.DataFrame([{"date": dates[1], "security_id": sid, "cash_dividend": 10,
                             "share_multiplier": 1, "payment_date": dates[-1]}])
    prices, actions = normalize_frames(prices, actions)
    daily = simulate(prices, actions, dates, targets, costs)["daily"]
    assert daily.nav.tolist() == pytest.approx([1000, 1000, 1000])
    assert daily.receivables.iloc[1] == (50 if sid == "1000" else -50)
    assert daily.receivables.iloc[-1] == 0


def test_split_and_rights_subscription_accounting(market):
    prices, _, dates, targets, costs = market
    prices.loc[(prices.security_id == "1000") & (prices.date >= dates[1]), ["open", "close", "high", "low"]] = [50, 50, 51, 49]
    actions = pd.DataFrame([{"date": dates[1], "security_id": "1000", "cash_dividend": 0,
                             "share_multiplier": 2}])
    prices, actions = normalize_frames(prices, actions)
    result = simulate(prices, actions, dates, targets, costs)
    assert result["daily"].nav.tolist() == pytest.approx([1000] * 3)
    assert result["holdings"].query("security_id == '1000'").quantity.tolist() == [5, 10, 10]
    actions.loc[0, ["share_multiplier", "subscription_ratio", "subscription_price"]] = [1.2, 0.2, 20]
    prices.loc[(prices.security_id == "1000") & (prices.date >= dates[1]), ["open", "close"]] = 104 / 1.2
    result = simulate(prices, actions, dates, targets, costs)
    assert result["daily"].nav.tolist() == pytest.approx([1000] * 3)


def test_failed_fill_is_not_redistributed_and_stale_holdings_persist(market):
    prices, actions, dates, targets, costs = market
    prices.loc[(prices.date == dates[0]) & (prices.security_id == "1000"), "buy_blocked"] = True
    result = simulate(prices, actions, dates, targets, costs)
    assert result["trades"].security_id.tolist() == ["1001"]
    assert result["daily"].gross_exposure.iloc[0] == pytest.approx(0.5)
    assert result["execution_diagnostics"].reason.iloc[0] == "adverse_limit_or_block"
    prices.loc[:, "buy_blocked"] = False
    prices = prices[~((prices.security_id == "1000") & (prices.date > dates[0]))]
    result = simulate(prices, actions, dates, targets, costs)
    assert result["daily"].nav.iloc[-1] == pytest.approx(1000)
    assert result["daily"].stale_positions.iloc[-1] == 1


def test_terminal_value_is_settled_and_reversal_charges_both_sides(market):
    prices, _, dates, targets, costs = market
    actions = pd.DataFrame([{"date": dates[1], "security_id": "1000", "terminal_value": 40}])
    prices, actions = normalize_frames(prices, actions)
    result = simulate(prices, actions, dates, targets, costs)
    assert result["daily"].nav.iloc[-1] == pytest.approx(700)
    assert result["daily"].long_pnl.sum() == pytest.approx(-300)
    _, no_actions = normalize_frames(prices)
    reversed_targets = targets.copy()
    reversed_targets["entry_date"] = dates[1]
    reversed_targets["weight"] *= -1
    result = simulate(prices, no_actions, dates, pd.concat([targets, reversed_targets]), costs)
    assert result["trades"].query("date == @dates[1]").notional.sum() == pytest.approx(2000)
    assert result["daily"].nav.iloc[-1] == pytest.approx(1000)


def test_adverse_open_limit_and_missing_exit_keep_positions(market):
    prices, actions, dates, targets, costs = market
    mask = (prices.security_id == "1000") & (prices.date == dates[0])
    prices.loc[mask, ["open", "high", "low", "close"]] = 110
    prices["limit_up"] = 110.0
    prices["limit_down"] = 90.0
    result = simulate(prices, actions, dates, targets, costs)
    assert result["trades"].security_id.tolist() == ["1001"]
    prices.loc[mask, ["open", "high", "low", "close"]] = [100, 101, 99, 100]
    prices.loc[(prices.date == dates[1]) & (prices.security_id == "1000"), "open"] = float("nan")
    closing = targets.copy()
    closing["entry_date"] = dates[1]
    closing["weight"] = 0.0
    result = simulate(prices, actions, dates, pd.concat([targets, closing]), costs)
    assert result["holdings"].query("date == @dates[1] and security_id == '1000'").quantity.iloc[0] == 5
    assert result["execution_diagnostics"].reason.tolist() == ["missing_open_or_opening_volume"]


def test_opening_fills_do_not_use_later_intraday_prices(market):
    prices, actions, dates, targets, costs = market
    original = simulate(prices, actions, dates[:1], targets, costs)["trades"]
    prices.loc[:, "high"] = 300.0
    prices.loc[:, "low"] = 20.0
    prices.loc[:, "close"] = 200.0
    prices.loc[:, "volume"] = 0.0
    changed = simulate(prices, actions, dates[:1], targets, costs)["trades"]
    pd.testing.assert_frame_equal(original, changed)
