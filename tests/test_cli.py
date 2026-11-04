import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from alpha_research import cli
from alpha_research.config import load_config
from alpha_research.provenance import file_checksums


def test_validation_history_preserves_reports_and_identifies_data(cached_research, monkeypatch, capsys):
    config_path = cached_research
    config_path.write_text(config_path.read_text().replace('runs = "runs"', 'runs = "artifacts/runs"'))
    config = load_config(config_path)
    instant = datetime(2026, 11, 4, 0, 0, tzinfo=timezone.utc)
    monkeypatch.setattr(cli, "datetime", SimpleNamespace(now=lambda tz: instant))
    elsewhere = config_path.parent / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    cli.main(["validate-data", "--config", str(config_path)])
    first_terminal = json.loads(capsys.readouterr().out)
    root = config_path.parent / "artifacts" / "validation"
    first = next(root.glob("validate-data-*.json"))
    original = first.read_bytes()
    record = json.loads(original)
    assert record["report"] == first_terminal
    assert record["created_at"] == instant.isoformat()
    assert record["config_path"] == str(config_path.resolve())
    assert record["config"] == config
    assert record["dataset"]["raw"] == config["paths"]["raw"]
    assert record["dataset"]["normalized"] == config["paths"]["normalized"]
    assert record["dataset"]["checksums"] == file_checksums(config["paths"]["normalized"])
    assert not (elsewhere / "artifacts").exists()

    master = Path(config["paths"]["raw"]) / "securities.csv"
    pd.read_csv(master).iloc[1:].to_csv(master, index=False)
    cli.main(["validate-data", "--config", str(config_path)])
    second_terminal = json.loads(capsys.readouterr().out)
    records = sorted(root.glob("*.json"))
    assert len(records) == 2  # Even with an identical timestamp.
    second = next(path for path in records if path != first)
    updated = json.loads(second.read_text())
    assert updated["report"] == second_terminal
    assert updated["dataset"]["fingerprint"] != record["dataset"]["fingerprint"]
    assert first.read_bytes() == original
    assert json.loads((Path(config["paths"]["normalized"]) / "coverage.json").read_text()) == second_terminal

    def fail(config):
        raise ValueError("Invalid input")
    monkeypatch.setattr(cli, "validate_data", fail)
    with pytest.raises(SystemExit) as error:
        cli.main(["validate-data", "--config", str(config_path)])
    assert error.value.code == 2
    assert len(list(root.iterdir())) == 2
