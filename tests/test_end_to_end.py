import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from alpha_research.cli import main
from alpha_research.config import load_config
from alpha_research.experiments import run


def test_offline_cli_reproduction_and_frozen_holdout(cached_research, monkeypatch):
    def forbid_network(*args, **kwargs):
        raise AssertionError("Offline acceptance test attempted a network request")
    monkeypatch.setattr("requests.Session.get", forbid_network)
    config_path = cached_research
    config = str(config_path)
    main(["fetch", "--config", config, "--offline"])
    main(["validate-data", "--config", config])
    main(["build-dataset", "--config", config])
    with pytest.raises(SystemExit) as error:
        main(["run", "--config", config, "--final-test"])
    assert error.value.code == 2
    main(["run", "--config", config, "--name", "validation-a"])
    main(["run", "--config", config, "--name", "validation-b"])
    root = config_path.parent / "runs"
    a, b = root / "validation-a", root / "validation-b"
    for filename in ["predictions.parquet", "trades.parquet", "daily.parquet", "targets.parquet"]:
        pd.testing.assert_frame_equal(pd.read_parquet(a / filename), pd.read_parquet(b / filename))
    assert json.loads((a / "metrics.json").read_text()) == json.loads((b / "metrics.json").read_text())
    main(["run", "--config", config, "--final-test", "--selection", str(a / "selection.json"), "--name", "test"])
    main(["report", "--config", config, "--run", str(root / "test")])
    predictions = pd.read_parquet(root / "test" / "predictions.parquet")
    assert predictions.date.min() >= pd.Timestamp("2022-01-01")
    assert pd.read_parquet(a / "predictions.parquet").date.max() < pd.Timestamp("2022-01-01")
    assert set(json.loads((a / "validation_scores.json").read_text())[i]["model"] for i in range(6)) == {
        "momentum", "ols", "ridge", "lasso", "elasticnet", "lightgbm"}
    assert (root / "test" / "performance.png").stat().st_size > 1000
    assert (root / "test" / "report.md").exists()
    assert "SYNTHETIC SOFTWARE DEMO" in (root / "test" / "report.md").read_text()
    assert json.loads((root / "test" / "coverage.json").read_text())["sources"] == ["SYNTHETIC_TEST"]
    manifest = json.loads((root / "test" / "manifest.json").read_text())
    assert manifest["config"]["research"]["seed"] == 103
    assert manifest["inputs"]["raw"]
    assert manifest["versions"]["lightgbm"]
    assert manifest["validation_run"] == "validation-a"
    assert set(manifest["code"]) == {"revision", "dirty"}
    assert not {"seed", "source_config_fingerprint", "selection"} & manifest.keys()
    selection = json.loads((a / "selection.json").read_text())
    assert set(selection) == {"model", "params", "validation_run", "config_fingerprint",
                              "code_fingerprint", "input_fingerprint"}
    assert manifest["input_fingerprint"] == selection["input_fingerprint"]
    selection_bytes = (a / "selection.json").read_bytes()
    daily = pd.read_parquet(root / "test" / "daily.parquet")
    rng = np.random.default_rng(103)
    factors = pd.DataFrame(rng.normal(0, 0.01, (len(daily), 4)), columns=["market", "smb", "hml", "momentum"])
    factors["date"], factors["rf"] = daily.date, 0.0
    factor_path = config_path.parent / "factors.csv"
    factors.to_csv(factor_path, index=False)
    main(["regress-factors", "--config", config, "--run", str(root / "test"),
          "--factors", str(factor_path), "--columns", "market", "smb", "hml", "momentum"])
    assert (root / "test" / "factor_regression" / "summary.json").exists()
    assert "Common-factor return attribution" in (root / "test" / "report.md").read_text()
    assert (a / "selection.json").read_bytes() == selection_bytes
    changed = load_config(config_path)
    changed["costs"]["borrow_rate"] = 0.9
    with pytest.raises(ValueError, match="Dataset config_fingerprint changed"):
        run(changed, final_test=True, selection_path=a / "selection.json")
    sample_path = Path(changed["paths"]["dataset"]) / "samples.parquet"
    samples = pd.read_parquet(sample_path)
    samples.loc[0, "target"] = 0.12345
    samples.to_parquet(sample_path, index=False)
    with pytest.raises(ValueError, match="Frozen input_fingerprint changed"):
        run(load_config(config_path), final_test=True, selection_path=a / "selection.json")
