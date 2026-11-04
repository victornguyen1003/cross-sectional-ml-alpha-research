"""Read and validate the research TOML; paths resolve relative to its file."""

import tomllib
from pathlib import Path

import pandas as pd


def load_config(path):
    path = Path(path).resolve()
    config = tomllib.loads(path.read_text())
    for key, value in config["paths"].items():
        config["paths"][key] = str((path.parent / value).resolve())
    r = config["research"]
    if not (pd.Timestamp(r["start"]) < pd.Timestamp(r["train_start"]) <
            pd.Timestamp(r["validation_start"]) < pd.Timestamp(r["test_start"]) <= pd.Timestamp(r["end"])):
        raise ValueError("Require start < train_start < validation_start < test_start <= end")
    for key in ["horizon", "rebalance", "min_history", "liquidity_window", "universe_size"]:
        if not isinstance(r[key], int) or r[key] < 1:
            raise ValueError(f"research.{key} must be a positive integer")
    if not 0 < r["decile"] <= 0.5:
        raise ValueError("decile must be in (0, 0.5]")
    if {"search", "random_candidates"} & config["model"].keys():
        raise ValueError("Remove obsolete model.search/random_candidates; tuning always uses the full grid")
    if config["model"]["cv_splits"] < 1:
        raise ValueError("cv_splits must be positive")
    if any(value < 0 for key, value in config["costs"].items() if key != "initial_nav"):
        raise ValueError("Costs cannot be negative")
    if config["costs"]["initial_nav"] <= 0:
        raise ValueError("initial_nav must be positive")
    if config["fetch"]["retries"] < 1 or config["fetch"]["delay_seconds"] < 0:
        raise ValueError("Invalid fetch retries/delay")
    if "factors" not in config or "factors" not in config["paths"]:
        raise ValueError("Add paths.factors and the [factors] settings from configs/default.toml")
    factors = config["factors"]
    columns = factors["columns"]
    if (not columns or len(columns) != len(set(columns)) or
            not set(columns).issubset({"mkt", "market_equity", "be_me", "ret_12_1"})):
        raise ValueError("Select distinct JKP factors from mkt, market_equity, be_me, ret_12_1")
    if factors["weighting"] not in {"vw_cap", "vw", "ew"}:
        raise ValueError("factors.weighting must be vw_cap, vw or ew")
    if factors["risk_free_column"] != "rf" or factors["fx_max_age_days"] < 0:
        raise ValueError("Use risk_free_column = rf and nonnegative fx_max_age_days")
    if len(set(config["paths"].values())) != len(config["paths"]):
        raise ValueError("Input and output directories must be distinct")
    return config

