"""Post-backtest OLS attribution to supplied common-factor returns.

Fit R_portfolio - R_f = alpha + factors @ beta + residual. Factor columns
already represent excess market returns or zero-investment spreads.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .fetch import digest
from .provenance import code_fingerprint, write_json


def regress_factors(daily, factors, columns, return_column="return", risk_free_column="rf"):
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


def analyze_run(run_path, factor_path, columns, return_column="return", risk_free_column="rf"):
    """Save attribution with its source checksums without changing model selection."""
    root, factor_path = Path(run_path), Path(factor_path)
    daily_path = root / "daily.parquet"
    summary, observations = regress_factors(pd.read_parquet(daily_path), pd.read_csv(factor_path),
                                           columns, return_column, risk_free_column)
    out = root / "factor_regression"
    out.mkdir(exist_ok=False)
    summary["inputs"] = {"daily_sha256": digest(daily_path), "factor_file": str(factor_path.resolve()),
                         "factors_sha256": digest(factor_path), "code_fingerprint": code_fingerprint()}
    write_json(out / "summary.json", summary)
    observations.to_parquet(out / "observations.parquet", index=False)
    return out
