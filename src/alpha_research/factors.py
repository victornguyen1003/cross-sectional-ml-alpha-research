"""Cached factor downloads and mandatory post-backtest return attribution.

Fit R_portfolio - R_f = alpha + factors @ beta + residual. Factor columns
already represent excess market returns or zero-investment spreads.
"""

import json
import re
import time
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import numpy as np
import pandas as pd
import requests

from .fetch import digest
from .provenance import code_fingerprint, write_json

JKP = "https://jkpfactors-data.s3.amazonaws.com/public"
FX_URL = "https://fred.stlouisfed.org/graph/fredgraph.csv?id=DEXTAUS"
RF_URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip"


def factor_inputs(config):
    """Verify the immutable download before fitting or reusing an experiment."""
    root = Path(config["paths"]["factors"])
    path = root / "factors.csv"
    if not (root / "factors.meta.json").is_file():
        raise FileNotFoundError("Missing mandatory factor data; run fetch-factors first")
    metadata = json.loads((root / "factors.meta.json").read_text())
    if metadata["settings"] != config["factors"] or metadata["currency"] != "USD":
        raise ValueError("Factor settings changed; use a fresh factor cache")
    if not path.is_file() or digest(path) != metadata["sha256"]:
        raise ValueError("Missing or corrupt factor cache")
    for name, source in metadata["sources"].items():
        if not (root / name).is_file() or digest(root / name) != source["sha256"]:
            raise ValueError(f"Missing or corrupt factor source cache: {name}")
    _factor_data(path.read_bytes(), config["factors"])
    return path, metadata


def _factor_data(body, settings):
    frame = pd.read_csv(BytesIO(body))
    required = ["date", *settings["columns"], settings["risk_free_column"], "fx_twd_per_usd"]
    if not set(required).issubset(frame):
        raise ValueError("Factor CSV is missing configured date, factor or risk-free columns")
    frame["date"] = pd.to_datetime(frame.date, errors="raise")
    if (frame.empty or frame.date.isna().any() or frame.date.duplicated().any() or
            frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all()):
        raise ValueError("Factor CSV requires unique, timezone-free daily session dates")
    frame[required[1:]] = frame[required[1:]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[required[1:]].to_numpy(dtype=float)).all():
        raise ValueError("Factor CSV contains missing or nonfinite returns")
    if (frame.fx_twd_per_usd <= 0).any():
        raise ValueError("FX rates must be positive Taiwan dollars per US dollar")
    return frame


def _download(root, name, url, settings, offline, session):
    path, meta = root / name, root / f"{name}.meta.json"
    if meta.exists():
        info = json.loads(meta.read_text())
        if info["url"] != url or not path.is_file() or digest(path) != info["sha256"]:
            raise ValueError(f"Missing, changed or corrupt factor source cache: {name}")
        return path.read_bytes(), info
    if offline:
        raise FileNotFoundError(f"No cached factor source: {name}; run fetch-factors first")
    if path.exists():
        raise ValueError(f"Incomplete factor cache; inspect and remove before retrying: {name}")
    last = None
    for attempt in range(settings["retries"]):
        time.sleep(settings["delay_seconds"] * (attempt + 1))
        try:
            response = session.get(url, timeout=settings["timeout_seconds"])
            response.raise_for_status()
            break
        except requests.RequestException as error:
            last = error
    else:
        raise RuntimeError(f"Factor download failed after retries: {url}") from last
    root.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(response.content)
    info = {"url": url, "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": digest(path)}
    write_json(meta, info)
    return response.content, info


def _zip_csv(body):
    try:
        with ZipFile(BytesIO(body)) as archive:
            names = [name for name in archive.namelist() if name.lower().endswith(".csv")]
            if len(names) != 1:
                raise ValueError("Expected exactly one CSV in factor archive")
            return archive.read(names[0])
    except BadZipFile as error:
        raise ValueError("Invalid factor ZIP archive") from error


def fetch_factors(config, offline=False):
    """Cache JKP Taiwan USD factors, FRED FX and a monthly USD T-bill proxy."""
    root, settings = Path(config["paths"]["factors"]), config["factors"]
    if (root / "factors.meta.json").exists():
        path, _ = factor_inputs(config)
        return str(path)
    if (root / "factors.csv").exists():
        raise ValueError("Incomplete normalized factor cache; inspect before retrying")
    series, sources = [], {}
    with requests.Session() as session:
        for name in settings["columns"]:
            filename = f"{name}.zip"
            url = f"{JKP}/%5Btwn%5D_%5B{name}%5D_%5Bdaily%5D_%5B{settings['weighting']}%5D.zip"
            body, sources[filename] = _download(root, filename, url, config["fetch"], offline, session)
            frame = pd.read_csv(BytesIO(_zip_csv(body)), na_values=["na"])
            if not {"location", "name", "freq", "weighting", "date", "ret"}.issubset(frame):
                raise ValueError(f"Unexpected JKP factor columns: {name}")
            if (not frame.location.eq("twn").all() or not frame.name.eq(name).all() or
                    not frame.freq.eq("daily").all() or not frame.weighting.eq(settings["weighting"]).all()):
                raise ValueError(f"Unexpected JKP country, factor, frequency or weighting: {name}")
            frame["date"] = pd.to_datetime(frame.date)
            if frame.date.isna().any() or frame.date.duplicated().any():
                raise ValueError(f"JKP factor requires unique dates: {name}")
            series.append(frame.set_index("date").ret.rename(name))
        fx_body, sources["fx.csv"] = _download(root, "fx.csv", FX_URL, config["fetch"], offline, session)
        rf_body, sources["risk_free.zip"] = _download(root, "risk_free.zip", RF_URL, config["fetch"], offline, session)
    factors = pd.concat(series, axis=1, sort=True).sort_index()
    factors = factors.loc[config["research"]["start"]:config["research"]["end"]]
    excluded_dates = factors.index[factors.isna().any(axis=1)].strftime("%Y-%m-%d").tolist()
    factors = factors.dropna()  # Record provider-calendar gaps; required portfolio sessions are checked separately.
    fx = pd.read_csv(BytesIO(fx_body), na_values=["."]).rename(columns={"observation_date": "date", "DEXTAUS": "fx_twd_per_usd"})
    if not {"date", "fx_twd_per_usd"}.issubset(fx):
        raise ValueError("Unexpected FRED FX columns")
    fx["date"] = pd.to_datetime(fx.date)
    if fx.date.isna().any() or fx.date.duplicated().any() or not fx.date.is_monotonic_increasing:
        raise ValueError("FX observations require unique sorted dates")
    fx["fx_twd_per_usd"] = pd.to_numeric(fx.fx_twd_per_usd, errors="raise")
    fx = fx.dropna(subset=["fx_twd_per_usd"]).rename(columns={"date": "fx_date"})
    factors = pd.merge_asof(factors.reset_index(), fx, left_on="date", right_on="fx_date",
                            direction="backward", tolerance=pd.Timedelta(days=settings["fx_max_age_days"]))
    # French monthly returns are percent; use the documented monthly/21 daily proxy.
    rows = [line.split(",") for line in _zip_csv(rf_body).decode("utf-8-sig").splitlines()
            if re.match(r"^\d{6},", line)]
    monthly = pd.DataFrame(rows, columns=["month", "market", "smb", "hml", "rf"])
    if monthly.month.duplicated().any():
        raise ValueError("Duplicate risk-free months")
    rf = pd.to_numeric(monthly.rf, errors="raise")
    if (rf <= -99).any():
        raise ValueError("Missing risk-free rate sentinel")
    monthly_rf = pd.Series(rf.to_numpy() / 100 / 21, index=monthly.month)
    factors[settings["risk_free_column"]] = factors.date.dt.strftime("%Y%m").map(monthly_rf)
    body = factors.to_csv(index=False).encode()
    _factor_data(body, settings)
    path = root / "factors.csv"
    with path.open("xb") as handle:
        handle.write(body)
    write_json(root / "factors.meta.json", {
        "url": "https://jkpfactors.com/data", "settings": settings, "currency": "USD", "units": "decimal",
        "retrieved_at": datetime.now(timezone.utc).isoformat(), "sha256": digest(path), "sources": sources,
        "excluded_incomplete_factor_dates": excluded_dates,
        "risk_free_convention": "Kenneth French monthly USD RF / 100 / 21; proxy for JKP's T30RET-based rate",
        "fx_convention": "FRED DEXTAUS, TWD per USD; last observed fixing within configured age; New York noon",
    })
    return str(path)


def usd_returns(daily, factors, previous_date, return_column="return"):
    """Convert TWD NAV returns to unhedged USD over consecutive portfolio sessions."""
    daily = daily.copy().sort_values("date")
    daily["date"] = pd.to_datetime(daily.date)
    previous_date = pd.Timestamp(previous_date)
    if previous_date >= daily.date.iloc[0]:
        raise ValueError("USD conversion requires an FX baseline before the first portfolio session")
    fx = factors.copy()
    fx["date"] = pd.to_datetime(fx.date)
    fx = fx.set_index("date").fx_twd_per_usd
    dates = pd.DatetimeIndex([previous_date, *daily.date])
    rates = fx.reindex(dates).to_numpy(dtype=float)
    if not np.isfinite(rates).all() or (rates <= 0).any():
        raise ValueError("Missing or invalid FX fixing for a portfolio session or its initial baseline")
    daily["portfolio_return_twd"] = daily[return_column]
    daily["fx_previous_twd_per_usd"], daily["fx_twd_per_usd"] = rates[:-1], rates[1:]
    daily[return_column] = (1 + daily[return_column]) * rates[:-1] / rates[1:] - 1
    return daily


def prepare_attribution(config, final_test=False):
    """Check required session and baseline coverage before training any models."""
    path, source = factor_inputs(config)
    factors = _factor_data(path.read_bytes(), config["factors"])
    calendar = pd.DatetimeIndex(pd.read_parquet(Path(config["paths"]["normalized"]) / "calendar.parquet").date)
    research = config["research"]
    start = pd.Timestamp(research["test_start"] if final_test else research["validation_start"])
    end = pd.Timestamp(research["end"]) if final_test else pd.Timestamp(research["test_start"]) - pd.Timedelta(days=1)
    dates = calendar[(calendar >= start) & (calendar <= end)]
    if dates.empty or len(dates) <= len(config["factors"]["columns"]) + 1:
        raise ValueError("Insufficient portfolio sessions for mandatory factor attribution")
    previous_date = calendar[calendar < dates.min()].max()
    if pd.isna(previous_date) or len(pd.DatetimeIndex([previous_date, *dates]).difference(factors.date)):
        raise ValueError("Mandatory factor attribution requires every portfolio session and its FX baseline")
    return path, source, previous_date


def regress_factors(daily, factors, columns, return_column="return", risk_free_column="rf",
                    require_complete=False):
    """Return OLS point estimates and the exact aligned regression sample.

    Inputs are daily decimal returns in the same currency and trading-session
    convention. Missing dates/values are dropped jointly, never forward-filled.
    Pass risk_free_column=None explicitly to assume a zero risk-free return.
    No standard errors, significance tests or causal interpretation are supplied.
    """
    columns = list(columns)
    if not columns or len(set(columns)) != len(columns):
        raise ValueError("Provide distinct, nonempty factor columns")
    if {"date", "portfolio_return", risk_free_column} & set(columns):
        raise ValueError("Factor columns cannot be date, portfolio_return or the risk-free column")
    daily, factors = daily.copy(), factors.copy()
    for frame in [daily, factors]:
        if "date" not in frame:
            raise ValueError("Each input requires a date column")
        frame["date"] = pd.to_datetime(frame.date, errors="raise")
        if frame.date.isna().any() or frame.date.duplicated().any():
            raise ValueError("Each input requires unique, nonmissing dates")
        if frame.date.dt.tz is not None or not frame.date.eq(frame.date.dt.normalize()).all():
            raise ValueError("Use timezone-free daily session dates")
    required = columns + ([risk_free_column] if risk_free_column is not None else [])
    if return_column not in daily or not set(required).issubset(factors):
        raise ValueError("Missing portfolio return, factor or risk-free column")
    sample = daily.set_index("date")[[return_column]].rename(columns={return_column: "portfolio_return"})
    sample = sample.join(factors.set_index("date")[required], how="inner").sort_index()
    matched_dates = len(sample)
    sample = sample.apply(pd.to_numeric, errors="raise")
    if np.isinf(sample.to_numpy(dtype=float)).any():
        raise ValueError("Infinite return observation")
    sample = sample.dropna()
    if require_complete and len(sample) != len(daily):
        raise ValueError("Mandatory factor attribution requires complete coverage of every portfolio session")
    if len(sample) <= len(columns) + 1:
        raise ValueError("Insufficient complete dates for factor regression")
    risk_free = sample[risk_free_column] if risk_free_column is not None else 0.0
    y = (sample.portfolio_return - risk_free).to_numpy(dtype=float)
    x = np.column_stack([np.ones(len(sample)), sample[columns].to_numpy(dtype=float)])
    coefficients, _, rank, _ = np.linalg.lstsq(x, y, rcond=None)
    if rank < x.shape[1]:
        raise ValueError("Factor design is rank deficient; remove constant or collinear factors")
    fitted = x @ coefficients
    residual = y - fitted
    squared_error = float(residual @ residual)
    total_variation = float(((y - y.mean()) ** 2).sum())
    summary = {
        "method": "OLS point estimates; no standard errors or significance tests",
        "return_column": return_column, "factor_columns": columns,
        "risk_free_column": risk_free_column, "zero_risk_free_assumed": risk_free_column is None,
        "portfolio_dates": len(daily), "matched_dates": matched_dates,
        "missing_factor_dates": len(daily) - matched_dates,
        "incomplete_matched_dates": matched_dates - len(sample), "observations": len(sample),
        "start": str(sample.index.min().date()), "end": str(sample.index.max().date()),
        "alpha_daily": float(coefficients[0]),
        "alpha_annualized_arithmetic": float(coefficients[0] * 252),
        "betas": dict(zip(columns, coefficients[1:].tolist())),
        "r_squared": 1 - squared_error / total_variation if total_variation > 0 else None,
        "residual_volatility_annualized": float(residual.std(ddof=1) * np.sqrt(252)),
    }
    # Prefix factor observations so their names cannot collide with diagnostics.
    observations = pd.DataFrame({"date": sample.index, "portfolio_return": sample.portfolio_return.to_numpy(),
                                 "portfolio_excess_return": y, "fitted_excess_return": fitted,
                                 "residual": residual})
    for column in columns:
        observations[f"factor_{column}"] = sample[column].to_numpy()
    return summary, observations


def analyze_run(run_path, factor_path, columns, return_column="return", risk_free_column="rf",
                source=None, previous_date=None):
    """Save attribution with its source checksums without changing model selection."""
    root, factor_path = Path(run_path), Path(factor_path)
    daily_path = root / "daily.parquet"
    daily, factors = pd.read_parquet(daily_path), pd.read_csv(factor_path)
    if source is not None and source["currency"] == "USD":
        if previous_date is None:
            raise ValueError("USD attribution requires the previous portfolio-session date")
        daily = usd_returns(daily, factors, previous_date, return_column)
    summary, observations = regress_factors(daily, factors,
                                           columns, return_column, risk_free_column, require_complete=True)
    out = root / "factor_regression"
    out.mkdir(exist_ok=False)
    summary["inputs"] = {"daily_sha256": digest(daily_path), "factor_file": str(factor_path.resolve()),
                         "factors_sha256": digest(factor_path), "code_fingerprint": code_fingerprint()}
    if source is not None:
        summary["source"] = source
        summary["currency"] = source["currency"]
    if "portfolio_return_twd" in daily:
        summary["conversion"] = {"formula": "(1 + return_twd) * fx_previous / fx_current - 1",
                                 "initial_fx_date": str(pd.Timestamp(previous_date).date())}
        observations = observations.merge(daily[["date", "portfolio_return_twd", "fx_previous_twd_per_usd",
                                                  "fx_twd_per_usd"]], on="date", validate="one_to_one")
    observations.to_parquet(out / "observations.parquet", index=False)
    summary["inputs"]["observations_sha256"] = digest(out / "observations.parquet")
    write_json(out / "summary.json", summary)
    return out


def attribution_summary(run_path):
    """Reject missing, incomplete or stale attribution before rendering a report."""
    root = Path(run_path)
    out = root / "factor_regression"
    if not (out / "summary.json").is_file() or not (out / "observations.parquet").is_file():
        raise ValueError("Mandatory factor attribution is missing; complete attribution before reporting")
    summary = json.loads((out / "summary.json").read_text())
    inputs = summary["inputs"]
    if (inputs["daily_sha256"] != digest(root / "daily.parquet") or
            inputs.get("observations_sha256") != digest(out / "observations.parquet")):
        raise ValueError("Factor attribution is stale or corrupt; cannot generate report")
    if summary["observations"] != summary["portfolio_dates"]:
        raise ValueError("Mandatory factor attribution has incomplete session coverage")
    return summary
