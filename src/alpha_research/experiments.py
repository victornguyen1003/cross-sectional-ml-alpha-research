"""Run one validation experiment or evaluate a frozen final-test specification."""

import pickle
from pathlib import Path

import pandas as pd

from .evaluation import report, save_evaluation
from .factors import analyze_run, prepare_attribution
from .models import fit_period
from .provenance import prepare_run, run_directory, save_metadata


def run(config, final_test=False, selection_path=None, name=None):
    snapshot, frozen = prepare_run(config, final_test, selection_path)
    factor_path, factor_source, previous_date = prepare_attribution(config, final_test)
    samples = pd.read_parquet(Path(config["paths"]["dataset"]) / "samples.parquet")
    result = fit_period(samples, config["research"], config["model"], frozen)
    out = run_directory(config, result["period"], name)
    save_evaluation(out, config, result)
    analyze_run(out, factor_path, config["factors"]["columns"],
                risk_free_column=config["factors"]["risk_free_column"], source=factor_source,
                previous_date=previous_date)
    for model_name, model in result["models"].items():
        if model is not None:
            with (out / f"model_{model_name}.pkl").open("wb") as handle:
                pickle.dump(model, handle, protocol=5)
    save_metadata(out, config, snapshot, frozen, result)
    report(out)
    return out
