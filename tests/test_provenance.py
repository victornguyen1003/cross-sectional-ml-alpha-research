import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from alpha_research import provenance


@pytest.fixture
def recorded_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(provenance, "code_fingerprint", lambda: "source-before")
    paths = {name: str(tmp_path / name) for name in ["raw", "normalized", "dataset", "runs", "factors"]}
    for name in ["raw", "normalized", "dataset", "factors"]:
        Path(paths[name]).mkdir()
        (Path(paths[name]) / "input.txt").write_text(f"{name} input")
    nested = Path(paths["normalized"]) / "nested" / "calendar.txt"
    nested.parent.mkdir()
    nested.write_text("2020-01-02")
    config = {"paths": paths, "research": {"seed": 103}}
    provenance.write_json(Path(paths["dataset"]) / "dataset.json", {
        "config_fingerprint": provenance.config_fingerprint(config), "code_fingerprint": "source-before",
        "normalized_checksums": provenance.file_checksums(paths["normalized"]),
    })
    snapshot, _ = provenance.prepare_run(config)
    selection = {"model": "ridge", "params": {"alpha": 1.0}, "validation_run": "validation-original",
                 **snapshot["fingerprints"]}
    selection_path = tmp_path / "selection.json"
    provenance.write_json(selection_path, selection)
    return config, selection_path


def test_matching_selection_and_relative_nested_checksums(recorded_inputs):
    config, selection_path = recorded_inputs
    snapshot, frozen = provenance.prepare_run(config, final_test=True, selection_path=selection_path)
    assert frozen["model"] == "ridge"
    assert "nested/calendar.txt" in snapshot["inputs"]["normalized"]
    assert provenance.fingerprint(snapshot["inputs"]) == snapshot["fingerprints"]["input_fingerprint"]


@pytest.mark.parametrize("changed", ["configuration", "source", "raw", "normalized", "dataset", "factors"])
def test_changed_research_inputs_block_final_test(recorded_inputs, monkeypatch, changed):
    config, selection_path = recorded_inputs
    if changed == "configuration":
        config["research"]["seed"] = 104
    elif changed == "source":
        monkeypatch.setattr(provenance, "code_fingerprint", lambda: "source-after")
    else:
        (Path(config["paths"][changed]) / "input.txt").write_text("changed")
    message = "Dataset .* changed" if changed in {"configuration", "source", "normalized"} else "Frozen input_fingerprint changed"
    with pytest.raises(ValueError, match=message):
        provenance.prepare_run(config, final_test=True, selection_path=selection_path)


@pytest.mark.parametrize("key", ["config_fingerprint", "code_fingerprint", "input_fingerprint"])
def test_frozen_fingerprint_mismatch_rejected(recorded_inputs, key):
    config, selection_path = recorded_inputs
    frozen = json.loads(selection_path.read_text())
    frozen[key] = "mismatch"
    provenance.write_json(selection_path, frozen)
    with pytest.raises(ValueError, match=f"Frozen {key} changed"):
        provenance.prepare_run(config, final_test=True, selection_path=selection_path)


@pytest.mark.parametrize("payload", [None, [], "not a selection"])
def test_nonobject_selection_cannot_bypass_final_test(recorded_inputs, payload):
    config, selection_path = recorded_inputs
    provenance.write_json(selection_path, payload)
    with pytest.raises(ValueError, match="JSON object"):
        provenance.prepare_run(config, final_test=True, selection_path=selection_path)


def test_final_test_flag_and_selection_must_agree(recorded_inputs):
    config, selection_path = recorded_inputs
    with pytest.raises(ValueError, match="requires --selection"):
        provenance.prepare_run(config, final_test=True)
    with pytest.raises(ValueError, match="only used with --final-test"):
        provenance.prepare_run(config, selection_path=selection_path)


def test_snapshot_is_computed_once_and_reused_for_metadata(recorded_inputs, monkeypatch):
    config, _ = recorded_inputs
    calls = Counter()
    original_digest = provenance.digest
    for name in ["config_fingerprint", "code_fingerprint"]:
        original = getattr(provenance, name)
        def counted(*args, _name=name, _original=original):
            calls[_name] += 1
            return _original(*args)
        monkeypatch.setattr(provenance, name, counted)
    def counted_digest(path):
        calls[str(path)] += 1
        return original_digest(path)
    monkeypatch.setattr(provenance, "digest", counted_digest)
    snapshot, _ = provenance.prepare_run(config)
    assert calls["config_fingerprint"] == calls["code_fingerprint"] == 1
    for group in snapshot["inputs"].values():
        assert group
    assert all(count == 1 for count in calls.values())
    before = calls.copy()
    # Even if inputs change after preparation, saving records the captured state
    # rather than silently replacing its fingerprints with a newer snapshot.
    (Path(config["paths"]["raw"]) / "input.txt").write_text("changed after preparation")
    out = provenance.run_directory(config, "validation", "validation-new")
    provenance.save_metadata(out, config, snapshot, None, {"model": "ridge", "params": {}, "period": "validation"})
    assert calls == before
    manifest = json.loads((out / "manifest.json").read_text())
    selection = json.loads((out / "selection.json").read_text())
    assert manifest["inputs"] == snapshot["inputs"]
    assert manifest["code_fingerprint"] == snapshot["fingerprints"]["code_fingerprint"]
    assert manifest["validation_run"] == selection["validation_run"] == out.name
    assert set(selection) == {"model", "params", "validation_run", *snapshot["fingerprints"]}
    assert not {"seed", "source_config_fingerprint", "selection"} & manifest.keys()


def test_final_manifest_references_validation_without_copying_selection(recorded_inputs):
    config, selection_path = recorded_inputs
    snapshot, frozen = provenance.prepare_run(config, final_test=True, selection_path=selection_path)
    out = provenance.run_directory(config, "test", "final-test")
    provenance.save_metadata(out, config, snapshot, frozen, {"period": "test"})
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest["validation_run"] == "validation-original"
    assert manifest["input_fingerprint"] == frozen["input_fingerprint"]
    assert "selection" not in manifest
    assert not (out / "selection.json").exists()


@pytest.mark.parametrize("status,dirty", [("", False), (" M source.py\n", True), (None, None)])
def test_git_state_contains_only_revision_and_dirty_status(monkeypatch, status, dirty):
    def git(args, **kwargs):
        value = status if args[1] == "status" else "revision-123"
        return SimpleNamespace(stdout=value or "", returncode=0 if value is not None else 128)
    monkeypatch.setattr(provenance.subprocess, "run", git)
    assert provenance.repository_state() == {"revision": "revision-123", "dirty": dirty}


def test_strict_json_serialization_of_numpy_dates_and_nonfinite_values(tmp_path):
    path = tmp_path / "record.json"
    provenance.write_json(path, {
        "integer": np.int64(103), "boolean": np.bool_(True), "float": np.float32(0.5),
        "nonfinite": [float("nan"), np.float64("inf"), -float("inf")],
        "dates": (pd.Timestamp("2020-01-02"), np.datetime64("2020-01-03")),
    })
    assert json.loads(path.read_text()) == {
        "integer": 103, "boolean": True, "float": 0.5, "nonfinite": [None, None, None],
        "dates": ["2020-01-02 00:00:00", "2020-01-03"],
    }
    with pytest.raises(TypeError):
        provenance.write_json(path, {"unsupported": object()})


def test_run_directory_rejects_unsafe_names_and_overwrites(tmp_path):
    config = {"paths": {"runs": str(tmp_path)}}
    with pytest.raises(ValueError, match="safe path"):
        provenance.run_directory(config, "validation", "../escape")
    provenance.run_directory(config, "validation", "same-name")
    with pytest.raises(FileExistsError):
        provenance.run_directory(config, "validation", "same-name")
