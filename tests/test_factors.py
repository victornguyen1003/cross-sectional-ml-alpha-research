from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pandas as pd
import pytest
import requests

from alpha_research.evaluation import report
from alpha_research.factors import (FX_URL, analyze_run, attribution_summary, factor_inputs,
                                    fetch_factors, prepare_attribution, regress_factors, usd_returns)


@pytest.fixture
def download_client(monkeypatch, factor_downloads):
    calls = []
    def get(self, url, **kwargs):
        calls.append(url)
        return SimpleNamespace(content=factor_downloads[url], raise_for_status=lambda: None)
    monkeypatch.setattr("requests.Session.get", get)
    return calls


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


def test_mandatory_attribution_cannot_drop_sessions(factor_sample):
    daily, factors = factor_sample
    with pytest.raises(ValueError, match="complete coverage"):
        regress_factors(daily, factors.iloc[1:], ["market", "smb"], require_complete=True)
    factors.loc[0, "smb"] = np.nan
    with pytest.raises(ValueError, match="complete coverage"):
        regress_factors(daily, factors, ["market", "smb"], require_complete=True)


def test_download_normalization_and_verified_offline_replay(factor_config, download_client):
    path = Path(fetch_factors(factor_config))
    assert len(download_client) == 6
    frame = pd.read_csv(path)
    assert set(factor_config["factors"]["columns"]).issubset(frame)
    np.testing.assert_allclose(frame.rf, 0.0001)  # Monthly percent / 100 / 21.
    assert frame.fx_twd_per_usd.eq(30).all()
    original = path.read_bytes()
    assert fetch_factors(factor_config, offline=True) == str(path)
    assert len(download_client) == 6
    assert path.read_bytes() == original
    _, metadata = factor_inputs(factor_config)
    assert metadata["currency"] == "USD"
    assert len(metadata["sources"]) == 6


@pytest.mark.parametrize("name", ["factors.csv", "mkt.zip"])
def test_corrupt_factor_cache_is_rejected(factor_config, download_client, name):
    fetch_factors(factor_config)
    (Path(factor_config["paths"]["factors"]) / name).write_bytes(b"changed")
    with pytest.raises(ValueError, match="corrupt"):
        fetch_factors(factor_config, offline=True)


def test_missing_offline_cache_and_incomplete_writes_fail(factor_config):
    with pytest.raises(FileNotFoundError, match="No cached factor"):
        fetch_factors(factor_config, offline=True)
    root = Path(factor_config["paths"]["factors"])
    root.mkdir()
    (root / "mkt.zip").write_bytes(b"interrupted")
    with pytest.raises(ValueError, match="Incomplete"):
        fetch_factors(factor_config)


def test_download_transport_retries_are_bounded(factor_config, monkeypatch):
    attempts = []
    def fail(*args, **kwargs):
        attempts.append(1)
        raise requests.ConnectionError("unavailable")
    monkeypatch.setattr("requests.Session.get", fail)
    with pytest.raises(RuntimeError, match="after retries"):
        fetch_factors(factor_config)
    assert len(attempts) == factor_config["fetch"]["retries"]
    assert not Path(factor_config["paths"]["factors"]).exists()


@pytest.mark.parametrize("bad_zip", [True, False])
def test_invalid_provider_response_never_completes_factor_dataset(factor_config, factor_downloads,
                                                                 download_client, bad_zip):
    url = next(url for url in factor_downloads if "%5Bmkt%5D" in url)
    body = b"<html>upstream error</html>"
    if not bad_zip:
        buffer = BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("data.csv", "date,unexpected\n2020-01-01,1")
        body = buffer.getvalue()
    factor_downloads[url] = body
    with pytest.raises(ValueError, match="Invalid factor ZIP|Unexpected JKP factor columns"):
        fetch_factors(factor_config)
    assert not (Path(factor_config["paths"]["factors"]) / "factors.meta.json").exists()


def test_fx_uses_only_past_fixings_and_enforces_maximum_age(factor_config, factor_downloads, download_client):
    factor_config["research"].update(start="2018-01-08", end="2018-01-12")
    fx = pd.read_csv(BytesIO(factor_downloads[FX_URL]))
    fx = fx[~fx.observation_date.isin(["2018-01-08", "2018-01-09"])]
    fx.loc[fx.observation_date == "2018-01-10", "DEXTAUS"] = 40
    factor_downloads[FX_URL] = fx.to_csv(index=False).encode()
    frame = pd.read_csv(fetch_factors(factor_config))
    assert frame.iloc[0].fx_twd_per_usd == 30
    assert frame.iloc[0].fx_date == "2018-01-05"
    factor_config["paths"]["factors"] += "-strict"
    factor_config["factors"]["fx_max_age_days"] = 1
    with pytest.raises(ValueError, match="missing or nonfinite"):
        fetch_factors(factor_config)


def test_inconsistent_provider_calendars_are_recorded(factor_config, factor_downloads, download_client):
    url = next(url for url in factor_downloads if "ret_12_1" in url)
    with ZipFile(BytesIO(factor_downloads[url])) as archive:
        frame = pd.read_csv(archive.open("data.csv"))
    extra = frame.iloc[:1].copy()
    extra["date"] = "2020-01-04"
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("data.csv", pd.concat([frame, extra]).to_csv(index=False))
    factor_downloads[url] = buffer.getvalue()
    result = pd.read_csv(fetch_factors(factor_config))
    assert "2020-01-04" not in set(result.date)
    _, metadata = factor_inputs(factor_config)
    assert metadata["excluded_incomplete_factor_dates"] == ["2020-01-04"]


def test_usd_conversion_uses_correct_quote_direction_and_session_baseline():
    dates = pd.to_datetime(["2020-01-03", "2020-01-06", "2020-01-07"])
    factors = pd.DataFrame({"date": dates, "fx_twd_per_usd": [30.0, 33.0, 30.0]})
    daily = pd.DataFrame({"date": dates[1:], "return": [0.1, -0.05]})
    original = daily.copy(deep=True)
    converted = usd_returns(daily, factors, dates[0])
    np.testing.assert_allclose(converted["return"], [0, 0.045], atol=1e-12)
    assert np.prod(1 + converted["return"]) == pytest.approx(np.prod(1 + daily["return"]) * 30 / 30)
    pd.testing.assert_frame_equal(daily, original)
    with pytest.raises(ValueError, match="baseline"):
        usd_returns(daily, factors.iloc[1:], dates[0])
    with pytest.raises(ValueError, match="before"):
        usd_returns(daily, factors, dates[1])


def test_required_coverage_is_checked_before_training(factor_config, download_client):
    fetch_factors(factor_config)
    root = Path(factor_config["paths"]["normalized"])
    root.mkdir()
    factor_config["research"].update(validation_start="2020-01-01", test_start="2021-01-01")
    dates = pd.bdate_range("2019-12-31", "2021-01-01").append(pd.DatetimeIndex(["2020-01-04"]))
    pd.DataFrame({"date": dates.sort_values()}).to_parquet(root / "calendar.parquet")
    with pytest.raises(ValueError, match="every portfolio session"):
        prepare_attribution(factor_config)


def test_report_requires_attribution_before_creating_outputs(tmp_path):
    with pytest.raises(ValueError, match="Mandatory factor attribution is missing"):
        report(tmp_path)
    assert not (tmp_path / "report.md").exists()
    assert not (tmp_path / "performance.png").exists()


@pytest.mark.parametrize("changed", ["daily.parquet", "factor_regression/observations.parquet"])
def test_report_rejects_stale_or_corrupt_attribution(tmp_path, factor_sample, changed):
    daily, factors = factor_sample
    daily.to_parquet(tmp_path / "daily.parquet", index=False)
    factors.to_csv(tmp_path / "factors.csv", index=False)
    analyze_run(tmp_path, tmp_path / "factors.csv", ["market", "smb"])
    assert attribution_summary(tmp_path)["observations"] == len(daily)
    (tmp_path / changed).write_bytes(b"changed")
    with pytest.raises(ValueError, match="stale or corrupt"):
        report(tmp_path)
