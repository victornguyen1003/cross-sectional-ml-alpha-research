"""Run one validation experiment or evaluate a frozen final-test specification."""

import pickle
from pathlib import Path

import pandas as pd

from .evaluation import report, save_evaluation
from .models import fit_period
from .provenance import prepare_run, run_directory, save_metadata


def run(config, final_test=False, selection_path=None, name=None):
    snapshot, frozen = prepare_run(config, final_test, selection_path)
    samples = pd.read_parquet(Path(config["paths"]["dataset"]) / "samples.parquet")
    result = fit_period(samples, config["research"], config["model"], frozen)
    out = run_directory(config, result["period"], name)
    save_evaluation(out, config, result)
    for model_name, model in result["models"].items():
        if model is not None:
            with (out / f"model_{model_name}.pkl").open("wb") as handle:
                pickle.dump(model, handle, protocol=5)
    save_metadata(out, config, snapshot, frozen, result)
    report(out)
    return out
