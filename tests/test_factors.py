import numpy as np
import pandas as pd
import pytest

from alpha_research.factors import regress_factors


@pytest.fixture
def factor_sample():
    dates = pd.bdate_range("2020-01-01", periods=30)
    rng = np.random.default_rng(103)
    factors = pd.DataFrame({"date": dates, "market": rng.normal(0, 0.01, len(dates)),
                            "smb": rng.normal(0, 0.005, len(dates)), "rf": 0.0002})
    daily = pd.DataFrame({"date": dates,
                          "return": factors.rf + 0.0007 + 1.2 * factors.market - 0.4 * factors.smb})
    return daily, factors


def test_known_alpha_loadings_and_risk_free_adjustment(factor_sample):
    daily, factors = factor_sample
    original = factors.copy(deep=True)
    summary, observations = regress_factors(daily, factors.sample(frac=1, random_state=103), ["market", "smb"])
    assert summary["alpha_daily"] == pytest.approx(0.0007, abs=1e-12)
    assert summary["alpha_annualized_arithmetic"] == pytest.approx(0.0007 * 252)
    assert summary["betas"] == pytest.approx({"market": 1.2, "smb": -0.4})
    assert summary["r_squared"] == pytest.approx(1)
    assert observations.residual.abs().max() < 1e-12
    assert observations.date.is_monotonic_increasing
    pd.testing.assert_frame_equal(factors, original)


def test_exact_date_alignment_and_complete_case_counts(factor_sample):
    daily, factors = factor_sample
    missing_date = factors.date.iloc[0]
    factors = factors.iloc[1:].copy()
    factors.loc[factors.index[0], "smb"] = np.nan
    summary, observations = regress_factors(daily, factors, ["market", "smb"])
    assert summary["missing_factor_dates"] == 1
    assert summary["incomplete_matched_dates"] == 1
    assert summary["observations"] == 28
    assert missing_date not in set(observations.date)


def test_zero_risk_free_is_explicit(factor_sample):
    daily, factors = factor_sample
    with pytest.raises(ValueError, match="Missing"):
        regress_factors(daily, factors.drop(columns="rf"), ["market", "smb"])
    summary, _ = regress_factors(daily, factors.drop(columns="rf"), ["market", "smb"], risk_free_column=None)
    assert summary["zero_risk_free_assumed"]
    assert summary["alpha_daily"] == pytest.approx(0.0009)


@pytest.mark.parametrize("which", [0, 1])
def test_duplicate_dates_rejected(factor_sample, which):
    inputs = list(factor_sample)
    inputs[which] = pd.concat([inputs[which], inputs[which].iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        regress_factors(*inputs, ["market", "smb"])


def test_collinear_factors_and_insufficient_sample_rejected(factor_sample):
    daily, factors = factor_sample
    factors["duplicate_market"] = factors.market * 2
    with pytest.raises(ValueError, match="rank deficient"):
        regress_factors(daily, factors, ["market", "duplicate_market"])
    with pytest.raises(ValueError, match="Insufficient"):
        regress_factors(daily.iloc[:3], factors.iloc[:3], ["market", "smb"])


def test_nonfinite_observations_rejected(factor_sample):
    daily, factors = factor_sample
    factors.loc[0, "market"] = np.inf
    with pytest.raises(ValueError, match="Infinite"):
        regress_factors(daily, factors, ["market", "smb"])
