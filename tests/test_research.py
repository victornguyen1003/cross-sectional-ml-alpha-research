import numpy as np
import pandas as pd
import pytest

from alpha_research.data import market_panel, normalize_frames
from alpha_research.features import FEATURES, build_features
from alpha_research.labels import build_labels
from alpha_research.models import make_model, purged_folds
from alpha_research.portfolio import target_weights
from alpha_research.universe import select_universe


def test_future_mutations_cannot_change_features_or_universe():
    dates = pd.bdate_range("2019-01-01", periods=330)
    prices = pd.DataFrame([{"date": date, "security_id": sid, "open": 100 + i,
                           "close": 100 + i, "high": 101 + i, "low": 99 + i,
                           "volume": 1000, "value": (100 + i) * 1000}
                          for i, date in enumerate(dates) for sid in ["1000", "1001"]])
    settings = {"min_history": 252, "liquidity_window": 60, "universe_size": 1}
    def compute(frame):
        frame, actions = normalize_frames(frame)
        return build_features(select_universe(market_panel(frame, actions, dates), settings))
    before = compute(prices)
    prices.loc[prices.date > dates[290], ["open", "close", "high", "low", "value", "volume"]] *= 5
    after = compute(prices)
    columns = FEATURES + ["selected", "eligible", "liquidity"]
    pd.testing.assert_frame_equal(before.loc[before.date <= dates[290], columns],
                                  after.loc[after.date <= dates[290], columns])
    # A larger identifier cannot win a tied historical liquidity ranking.
    assert before.loc[before.selected, "security_id"].eq("1000").all()


def test_labels_use_next_open_and_full_twenty_session_horizon():
    dates = pd.bdate_range("2020-01-01", periods=50)
    panel = pd.DataFrame({"date": dates, "security_id": "1000", "selected": True,
                          "total_open": np.arange(1, 51, dtype=float)})
    samples = build_labels(panel, dates, {"train_start": "2020-01-01", "horizon": 20, "rebalance": 20})
    assert samples.iloc[0].entry_date == dates[1]
    assert samples.iloc[0].label_end == dates[21]
    assert samples.iloc[0].realized_return == pytest.approx(22 / 2 - 1)
    assert not samples.iloc[-1].label_available


def test_purged_folds_group_dates_and_remove_overlapping_labels():
    dates = pd.bdate_range("2019-01-01", periods=30)
    samples = pd.DataFrame([{"date": d, "label_end": d + pd.offsets.BDay(5), "security_id": sid}
                            for d in dates for sid in ["1000", "1001", "1002"]])
    for train_i, valid_i in purged_folds(samples, 3):
        train, valid = samples.iloc[train_i], samples.iloc[valid_i]
        assert train.label_end.max() < valid.date.min()
        assert not set(train.date) & set(valid.date)
        assert valid.groupby("date").size().eq(3).all()


def test_imputation_and_scaling_fit_training_only():
    train = pd.DataFrame(np.arange(60).reshape(6, 10), columns=FEATURES, dtype=float)
    train.iloc[0, 0] = np.nan
    model = make_model("ridge", {"alpha": 1}, 103)
    model.fit(train, np.linspace(0, 1, len(train)))
    before = model[0].statistics_.copy(), model[1].mean_.copy()
    model.predict(pd.DataFrame(1e9, index=[0], columns=FEATURES))
    assert np.array_equal(before[0], model[0].statistics_)
    assert np.array_equal(before[1], model[1].mean_)


def test_target_ties_are_deterministic_and_dollar_neutral():
    rows = pd.DataFrame({"date": pd.Timestamp("2020-01-01"), "entry_date": pd.Timestamp("2020-01-02"),
                         "security_id": [str(1000 + i) for i in range(20)], "prediction": 1.0})
    weights = target_weights(rows.sample(frac=1, random_state=103))
    assert weights.weight.sum() == pytest.approx(0)
    assert weights.weight.abs().sum() == pytest.approx(1)
    assert set(weights[weights.weight < 0].security_id) == {"1000", "1001"}
    assert set(weights[weights.weight > 0].security_id) == {"1018", "1019"}
