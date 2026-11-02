import pandas as pd
import pytest

from alpha_research.data import normalize_frames


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
