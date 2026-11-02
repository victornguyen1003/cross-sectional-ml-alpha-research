"""Small command-line interface; all paths are relative to the TOML file."""

import argparse
import json
from pathlib import Path

from .data import validate_data
from .evaluation import json_safe, report
from .experiments import build_dataset, load_config, run
from .fetch import fetch


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproducible exploratory TWSE alpha research")
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ["fetch", "validate-data", "build-dataset", "run", "report"]:
        command = commands.add_parser(name)
        command.add_argument("--config", required=True, type=Path)
        if name == "fetch":
            command.add_argument("--offline", action="store_true")
            command.add_argument("--bulk", action="store_true", help="Fetch all weekdays; defaults to representative probes")
        if name == "run":
            command.add_argument("--final-test", action="store_true")
            command.add_argument("--selection", type=Path)
            command.add_argument("--name")
        if name == "report":
            command.add_argument("--run", required=True, type=Path)
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
        else:
            result = str(report(args.run))
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        parser.exit(2, f"error: {error}\n")
    print(json.dumps(json_safe(result), indent=2))


if __name__ == "__main__":
    main()
