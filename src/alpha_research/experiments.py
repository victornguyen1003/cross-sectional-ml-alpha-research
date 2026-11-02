"""Configuration, dataset construction, frozen selection and run provenance."""

import hashlib
import importlib.metadata
import json
import pickle
import re
import subprocess
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .backtest import simulate
from .data import market_panel
from .evaluation import performance, predictive_metrics, report, sensitivity, write_json
from .features import FEATURES, build_features
from .fetch import digest
from .labels import build_labels
from .models import MODEL_ORDER, fit_predict, mean_rank_ic, tune_models
from .portfolio import target_weights
from .universe import select_universe


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
    if config["model"]["search"] not in {"grid", "random"}:
        raise ValueError("model.search must be grid or random")
    if config["model"]["cv_splits"] < 1:
        raise ValueError("cv_splits must be positive")
    if any(value < 0 for key, value in config["costs"].items() if key != "initial_nav"):
        raise ValueError("Costs cannot be negative")
    if config["costs"]["initial_nav"] <= 0:
        raise ValueError("initial_nav must be positive")
    if config["fetch"]["retries"] < 1 or config["fetch"]["delay_seconds"] < 0:
        raise ValueError("Invalid fetch retries/delay")
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def config_fingerprint(config):
    return fingerprint({key: value for key, value in config.items() if key != "paths"})


def code_fingerprint():
    files = sorted(Path(__file__).parent.glob("*.py"))
    lock = Path(__file__).resolve().parents[2] / "uv.lock"
    if lock.exists():
        files.append(lock)
    return fingerprint({file.name: digest(file) for file in files})


def input_checksums(config):
    result = {}
    for name in ["raw", "normalized", "dataset"]:
        root = Path(config["paths"][name])
        result[name] = {str(path.relative_to(root)): digest(path)
                        for path in sorted(root.rglob("*")) if path.is_file()}
    return result


def build_dataset(config):
    root, out = Path(config["paths"]["normalized"]), Path(config["paths"]["dataset"])
    out.mkdir(parents=True, exist_ok=True)
    prices = pd.read_parquet(root / "prices.parquet")
    actions = pd.read_parquet(root / "actions.parquet")
    calendar = pd.DatetimeIndex(pd.read_parquet(root / "calendar.parquet").date)
    panel = market_panel(prices, actions, calendar)
    panel = select_universe(panel, config["research"])
    panel = build_features(panel)
    samples = build_labels(panel, calendar, config["research"])
    if samples.empty:
        raise ValueError("No eligible samples; inspect history, security types and liquidity")
    samples.to_parquet(out / "samples.parquet", index=False)
    panel[["date", "security_id", "security_type", "identity_verified", "history_sessions",
           "liquidity", "eligible", "selected"]].to_parquet(out / "eligibility.parquet", index=False)
    summary = {"rows": len(samples), "signal_dates": samples.date.nunique(),
               "missing_labels": int((~samples.label_available).sum()), "features": FEATURES,
               "config_fingerprint": config_fingerprint(config),
               "code_fingerprint": code_fingerprint(),
               "normalized_checksums": {p.name: digest(p) for p in sorted(root.glob("*")) if p.is_file()}}
    write_json(out / "dataset.json", summary)
    return summary


def repository_state():
    def git(*args):
        process = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return process.stdout.strip() if process.returncode == 0 else None
    return {"revision": git("rev-parse", "HEAD"), "status": git("status", "--porcelain"),
            "diff_sha256": fingerprint(git("diff", "HEAD"))}


def run(config, final_test=False, selection_path=None, name=None):
    np.random.seed(config["research"]["seed"])
    paths, r = config["paths"], config["research"]
    dataset = Path(paths["dataset"])
    metadata = json.loads((dataset / "dataset.json").read_text())
    if metadata["config_fingerprint"] != config_fingerprint(config):
        raise ValueError("Configuration changed; rebuild the dataset")
    if metadata["code_fingerprint"] != code_fingerprint():
        raise ValueError("Research code changed; rebuild the dataset")
    current_normalized = {p.name: digest(p) for p in sorted(Path(paths["normalized"]).glob("*")) if p.is_file()}
    if metadata["normalized_checksums"] != current_normalized:
        raise ValueError("Normalized inputs changed; rebuild the dataset")
    checksums = input_checksums(config)
    frozen = None
    if final_test:
        if selection_path is None:
            raise ValueError("Final test requires --selection from a completed validation run")
        frozen = json.loads(Path(selection_path).read_text())
        if frozen["config_fingerprint"] != config_fingerprint(config):
            raise ValueError("Frozen specification does not match this configuration")
        if frozen["input_fingerprint"] != fingerprint(checksums):
            raise ValueError("Inputs changed since validation; final test is blocked")
        if frozen["code_fingerprint"] != code_fingerprint():
            raise ValueError("Research code changed since validation; rerun validation")
    elif selection_path is not None:
        raise ValueError("--selection is only used with --final-test")
    samples = pd.read_parquet(dataset / "samples.parquet")
    validation_start, test_start = pd.Timestamp(r["validation_start"]), pd.Timestamp(r["test_start"])
    train = samples[(samples.date >= pd.Timestamp(r["train_start"])) &
                    (samples.date < validation_start) & (samples.label_end < validation_start) &
                    samples.label_available].copy()
    if train.empty:
        raise ValueError("No mature initial training samples")
    fitted, validation_scores = {}, []
    if final_test:
        selected = frozen["model"]
        chosen_params = frozen["params"]
        fitting = samples[(samples.date >= pd.Timestamp(r["train_start"])) &
                          (samples.date < test_start) & (samples.label_end < test_start) &
                          samples.label_available]
        evaluation = samples[samples.date >= test_start].copy()
        model, predictions = fit_predict(selected, chosen_params, fitting, evaluation, r["seed"])
        fitted[selected] = model
        cv_scores = []
        period = "test"
    else:
        parameters, cv_scores = tune_models(train, config["model"], r["seed"])
        evaluation = samples[(samples.date >= validation_start) & (samples.date < test_start) &
                             (samples.entry_date < test_start)].copy()
        # Do not evaluate labels that extend into the final-test period.
        boundary = evaluation.label_end >= test_start
        evaluation.loc[boundary, ["realized_return", "target"]] = np.nan
        evaluation.loc[boundary, "label_available"] = False
        if evaluation.empty:
            raise ValueError("No validation samples")
        all_predictions = {}
        for model_name in MODEL_ORDER:
            model, predicted = fit_predict(model_name, parameters.get(model_name, {}), train,
                                           evaluation, r["seed"])
            fitted[model_name] = model
            all_predictions[model_name] = predicted
            validation_scores.append({"model": model_name, "rank_ic": mean_rank_ic(predicted)})
        finite = [score for score in validation_scores if np.isfinite(score["rank_ic"])]
        if not finite:
            raise ValueError("No usable validation rank correlations; cannot freeze selection")
        best_score = max(score["rank_ic"] for score in finite)
        selected = next(score["model"] for score in finite if score["rank_ic"] >= best_score - 1e-12)
        chosen_params = parameters.get(selected, {})
        predictions = all_predictions[selected]
        period = "validation"
    if evaluation.empty:
        raise ValueError("No evaluation samples")
    run_name = name or f"{period}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", run_name):
        raise ValueError("Run name must be a single safe path component")
    out = Path(paths["runs"]) / run_name
    out.mkdir(parents=True, exist_ok=False)
    normalized = Path(paths["normalized"])
    prices = pd.read_parquet(normalized / "prices.parquet")
    actions = pd.read_parquet(normalized / "actions.parquet")
    calendar = pd.DatetimeIndex(pd.read_parquet(normalized / "calendar.parquet").date)
    period_start = test_start if final_test else validation_start
    calendar = calendar[(calendar >= period_start) & ((calendar <= pd.Timestamp(r["end"])) if final_test else (calendar < test_start))]
    targets = target_weights(predictions, r["decile"])
    if targets.empty:
        raise ValueError("No portfolio targets; require enough finite scores for two deciles")
    ledger = simulate(prices, actions, calendar, targets, config["costs"])
    predictive, rank_ic, deciles = predictive_metrics(predictions)
    metrics = {"period": period, "model": selected, "params": chosen_params,
               "predictive": predictive, "portfolio": performance(ledger["daily"], config["costs"]["initial_nav"]),
               "sensitivity": sensitivity(prices, actions, calendar, targets, config["costs"], config["evaluation"])}
    for key, frame in ledger.items():
        frame.to_parquet(out / f"{key}.parquet", index=False)
    predictions.to_parquet(out / "predictions.parquet", index=False)
    targets.to_parquet(out / "targets.parquet", index=False)
    rank_ic.to_parquet(out / "rank_ic.parquet", index=False)
    deciles.to_parquet(out / "deciles.parquet", index=False)
    pd.read_parquet(dataset / "eligibility.parquet").query("date >= @period_start").loc[
        lambda f: f.date.isin(calendar)].to_parquet(out / "eligibility.parquet", index=False)
    if not final_test:
        for model_name, predicted in all_predictions.items():
            predicted.to_parquet(out / f"predictions_{model_name}.parquet", index=False)
    for model_name, model in fitted.items():
        if model is not None:
            with (out / f"model_{model_name}.pkl").open("wb") as handle:
                pickle.dump(model, handle, protocol=5)
    for benchmark in ["momentum", "equal_weight"]:
        benchmark_predictions = evaluation.copy()
        benchmark_predictions["prediction"] = evaluation.momentum if benchmark == "momentum" else 1.0
        weights = target_weights(benchmark_predictions, r["decile"], long_only=benchmark == "equal_weight")
        result = simulate(prices, actions, calendar, weights, config["costs"])
        result["daily"].to_parquet(out / f"benchmark_{benchmark}_daily.parquet", index=False)
        metrics[f"benchmark_{benchmark}"] = performance(result["daily"], config["costs"]["initial_nav"])
    coverage = json.loads((normalized / "coverage.json").read_text())
    coverage["missing_evaluation_labels"] = int(predictions.realized_return.isna().sum())
    coverage["notes"].extend([
        "Missing forward labels remain in predictions and trading, but cannot enter model fitting or labeled diagnostics.",
        "Incomplete-label cross-sections can bias training ranks and predictive diagnostics.",
        "Dividend targets use a reinvested total-return index; the ledger holds distributions as cash/receivables.",
        "Opening fills use supplied opening-time limits/blocks; missing auction liquidity and limit history remain unresolved.",
        "Targets use pre-fee NAV. Costs can cause small departures from target gross and net weights.",
    ])
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "pyarrow", "requests", "scikit-learn", "lightgbm", "matplotlib"]}
    manifest = {"created_at": datetime.now(timezone.utc).isoformat(), "config": config,
                "code": repository_state(), "code_fingerprint": code_fingerprint(), "versions": versions,
                "inputs": checksums, "period": period, "seed": r["seed"],
                "selection": frozen, "source_config_fingerprint": config_fingerprint(config)}
    write_json(out / "manifest.json", manifest)
    write_json(out / "metrics.json", metrics)
    write_json(out / "coverage.json", coverage)
    write_json(out / "cv_scores.json", cv_scores)
    write_json(out / "validation_scores.json", validation_scores)
    if not final_test:
        write_json(out / "selection.json", {"model": selected, "params": chosen_params,
                   "config_fingerprint": config_fingerprint(config), "input_fingerprint": fingerprint(checksums),
                   "code_fingerprint": code_fingerprint(), "validation_run": run_name,
                   "features": FEATURES, "validation_scores": validation_scores})
    report(out)
    return out
