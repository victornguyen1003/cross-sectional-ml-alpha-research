"""Small command-line interface; all paths are relative to the TOML file."""

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import build_dataset, validate_data
from .evaluation import report
from .experiments import run
from .fetch import fetch
from .factors import analyze_run
from .provenance import json_safe


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible exploratory TWSE alpha research")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["fetch", "validate-data", "build-dataset", "run", "report", "regress-factors"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name == "fetch":
            command.add_argument("--offline", action="store_true")
            command.add_argument("--bulk", action="store_true", help="Fetch all weekdays; defaults to representative probes")
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
        elif args.command == "validate-data":
            result = validate_data(config)
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
