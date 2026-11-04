"""Input/code fingerprints, frozen selection and run provenance."""

import hashlib
import importlib.metadata
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from .fetch import digest


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


def file_checksums(root):
    root = Path(root)
    return {str(path.relative_to(root)): digest(path)
            for path in sorted(root.rglob("*")) if path.is_file()}


def repository_state():
    def git(*args):
        process = subprocess.run(["git", *args], capture_output=True, text=True, check=False)
        return process.stdout.strip() if process.returncode == 0 else None
    status = git("status", "--porcelain")
    return {"revision": git("rev-parse", "HEAD"), "dirty": bool(status) if status is not None else None}


def json_safe(value):
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(value)
    return value


def write_json(path, payload):
    Path(path).write_text(json.dumps(json_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def prepare_run(config, final_test=False, selection_path=None):
    """Capture fingerprints once and reject stale datasets or frozen selections."""
    if final_test and selection_path is None:
        raise ValueError("Final test requires --selection from a completed validation run")
    if not final_test and selection_path is not None:
        raise ValueError("--selection is only used with --final-test")
    checksums = {name: file_checksums(config["paths"][name]) for name in ["raw", "normalized", "dataset"]}
    hashes = {"config_fingerprint": config_fingerprint(config), "code_fingerprint": code_fingerprint(),
              "input_fingerprint": fingerprint(checksums)}
    metadata = json.loads((Path(config["paths"]["dataset"]) / "dataset.json").read_text())
    expected = {"config_fingerprint": hashes["config_fingerprint"], "code_fingerprint": hashes["code_fingerprint"],
                "normalized_checksums": checksums["normalized"]}
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"Dataset {key} changed; rebuild the dataset")
    frozen = json.loads(Path(selection_path).read_text()) if final_test else None
    if final_test:
        if not isinstance(frozen, dict):
            raise ValueError("Frozen selection must be a JSON object")
        for key, value in hashes.items():
            if frozen.get(key) != value:
                raise ValueError(f"Frozen {key} changed since validation; rerun validation")
    return {"inputs": checksums, "fingerprints": hashes}, frozen


def run_directory(config, period, name=None):
    name = name or f"{period}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid4().hex[:8]}"
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", name):
        raise ValueError("Run name must be a single safe path component")
    out = Path(config["paths"]["runs"]) / name
    out.mkdir(parents=True, exist_ok=False)
    return out


def save_metadata(out, config, snapshot, frozen, result):
    """Write lean records using the fingerprints captured before fitting."""
    versions = {name: importlib.metadata.version(name) for name in [
        "numpy", "pandas", "pyarrow", "requests", "scikit-learn", "lightgbm", "matplotlib"]}
    hashes = snapshot["fingerprints"]
    validation_run = frozen["validation_run"] if frozen is not None else out.name
    write_json(out / "manifest.json", {
        "created_at": datetime.now(timezone.utc).isoformat(), "config": config,
        "code": repository_state(), "code_fingerprint": hashes["code_fingerprint"], "versions": versions,
        "inputs": snapshot["inputs"], "input_fingerprint": hashes["input_fingerprint"],
        "period": result["period"], "validation_run": validation_run,
    })
    if frozen is None:
        write_json(out / "selection.json", {
            "model": result["model"], "params": result["params"], "validation_run": validation_run, **hashes,
        })
