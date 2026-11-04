"""Small command-line interface; configured paths resolve relative to the TOML file."""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import load_config
from .data import build_dataset, validate_data
from .evaluation import report
from .experiments import run
from .fetch import fetch, fetch_actions, fetch_calendar, fetch_securities
from .factors import analyze_run, fetch_factors
from .provenance import file_checksums, fingerprint, json_safe


def save_validation_report(config, config_path, result):
    """Keep validation history outside the fingerprinted input directories."""
    created_at = datetime.now(timezone.utc)
    checksums = file_checksums(config["paths"]["normalized"])
    record = {
        "created_at": created_at.isoformat(), "command": "validate-data",
        "config_path": str(Path(config_path).resolve()), "config": config,
        "dataset": {"raw": config["paths"]["raw"], "normalized": config["paths"]["normalized"],
                    "checksums": checksums, "fingerprint": fingerprint(checksums)},
        "report": result,
    }
    serialized = json.dumps(json_safe(record), indent=2, sort_keys=True, allow_nan=False) + "\n"
    root = Path(config["paths"]["runs"]).parent / "validation"
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"validate-data-{created_at:%Y%m%dT%H%M%S%fZ}-{uuid4().hex[:8]}.json"
    with path.open("x", encoding="utf-8") as handle:
        handle.write(serialized)
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible exploratory TWSE alpha research")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["fetch", "fetch-calendar", "fetch-securities", "fetch-actions", "fetch-factors", "validate-data", "build-dataset", "run", "report", "regress-factors"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name in {"fetch", "fetch-calendar", "fetch-securities", "fetch-actions", "fetch-factors"}:
            command.add_argument("--offline", action="store_true")
        if name == "fetch":
            command.add_argument("--bulk", action="store_true", help="Fetch actual exchange sessions and historical action terms; defaults to representative probes")
        if name == "run":
            command.add_argument("--final-test", action="store_true")
            command.add_argument("--selection", type=Path)
            command.add_argument("--name")
        if name in {"report", "regress-factors"}:
            command.add_argument("--run", required=True, type=Path)
        if name == "regress-factors":
            command.add_argument("--factors", required=True, type=Path, help="CSV of daily decimal factor returns")
            command.add_argument("--columns", required=True, nargs="+")
            command.add_argument("--return-column", choices=["return", "gross_return"], default="return")
            risk_free = command.add_mutually_exclusive_group()
            risk_free.add_argument("--risk-free-column", default="rf")
            risk_free.add_argument("--zero-risk-free", action="store_true")
    args = parser.parse_args(argv)
    try:
        config = load_config(args.config)
        if args.command == "fetch":
            result = fetch(config, offline=args.offline, bulk=args.bulk)
        elif args.command == "fetch-calendar":
            result = {"sessions": len(fetch_calendar(config, offline=args.offline))}
        elif args.command == "fetch-securities":
            result = fetch_securities(config, offline=args.offline)
        elif args.command == "fetch-actions":
            result = fetch_actions(config, offline=args.offline)
        elif args.command == "fetch-factors":
            result = fetch_factors(config, offline=args.offline)
        elif args.command == "validate-data":
            result = validate_data(config)
            save_validation_report(config, args.config, result)
        elif args.command == "build-dataset":
            result = build_dataset(config)
        elif args.command == "run":
            result = str(run(config, final_test=args.final_test, selection_path=args.selection, name=args.name))
        elif args.command == "regress-factors":
            result = str(analyze_run(args.run, args.factors, args.columns, args.return_column,
                                    None if args.zero_risk_free else args.risk_free_column))
            report(args.run)
        else:
            result = str(report(args.run))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
