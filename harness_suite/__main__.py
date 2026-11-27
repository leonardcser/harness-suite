"""harness-suite: test coding agents by pretending to be their model provider.

python -m harness_suite list
python -m harness_suite harnesses
python -m harness_suite images [NAME ...]
python -m harness_suite run --quick --harness NAME
python -m harness_suite run [--harness NAME] [--check ID] [--api API] [--tools MODE] [--frontend NAME]
python -m harness_suite report results/RUN [--details]
python -m harness_suite export results/RUN [results/RERUN ...] OUTPUT.json
"""

import argparse
import json
import shlex
import signal
import sys
import time
from pathlib import Path

from . import images, limits, recipe, report, runner
from .results import REQUIRED
from .sandbox import image_exists
from .scenarios import all_checks, all_scenarios, find


def interrupt(number, frame) -> None:
    raise KeyboardInterrupt(signal.Signals(number).name)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="harness-suite", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    commands = parser.add_subparsers(dest="command", required=True)
    listing = commands.add_parser("list", help="list the checks")
    listing.add_argument("--json", action="store_true")
    commands.add_parser("harnesses", help="list the harness recipes")
    built = commands.add_parser("images", help="build the pinned image of each harness")
    built.add_argument("names", nargs="*", metavar="NAME", help="recipe names; default: every recipe with [install]")
    run = commands.add_parser("run", help="run checks against harnesses")
    run.add_argument("--harness", action="append", help="a recipe name or .toml path; repeatable")
    run.add_argument("--check", action="append", help="a check or scenario ID, or an area like tools; repeatable")
    run.add_argument("--api", action="append", choices=recipe.APIS)
    run.add_argument("--tools", action="append", choices=("mcp", "native"))
    run.add_argument("--frontend", action="append", choices=("headless", "tui"))
    run.add_argument(
        "--quick",
        action="store_true",
        help="only the required checks, once each, headless on the first API: a first result in minutes",
    )
    run.add_argument(
        "--host",
        action="store_true",
        help="run harnesses from PATH, not their images; a recipe without an image always does",
    )
    run.add_argument(
        "--command",
        dest="overrides",
        action="append",
        default=[],
        metavar="NAME=COMMAND",
        help="run one harness on the host with this command",
    )
    run.add_argument(
        "--jobs",
        type=int,
        default=limits.jobs(),
        help="most cases at once; each starts only while the machine has room (default: %(default)s)",
    )
    run.add_argument("--output", type=Path)
    shown = commands.add_parser("report", help="print a results grid")
    shown.add_argument("results", type=Path)
    shown.add_argument("--details", action="store_true", help="also explain every fail, gap and suite error")
    exported = commands.add_parser("export", help="write a run as one JSON file")
    exported.add_argument("runs", type=Path, nargs="+", metavar="RUN", help="runs, each case from the last that has it")
    exported.add_argument("output", type=Path)
    args = parser.parse_args()

    if args.command == "list":
        found = []
        for info, scenario in all_checks():
            variants = [v for v in scenario.variants if info.applies(v)] if scenario.id else ["default"]
            found.append({**vars(info), "scenario": scenario.id, "variants": variants})
        if args.json:
            print(json.dumps(found, indent=2))
            return 0
        for c in found:
            variants = "" if c["variants"] == ["default"] else " [" + ", ".join(c["variants"]) + "]"
            print(f"{c['level']:<12} {c['id']:<28} {c['title']}{variants}")
        return 0
    if args.command == "harnesses":
        for name in recipe.available():
            r = recipe.load(name)
            settings = ", ".join(sorted(r.data.get("settings", {})))
            present = "image ready" if r.image and image_exists(r.image) else "no image"
            print(f"{r.name:<12} {r.version:<10} {', '.join(r.apis):<28} {present:<12} settings: {settings}")
        return 0
    if args.command == "images":
        recipes = [recipe.load(name) for name in args.names or recipe.available()]
        if missing := [r.name for r in recipes if not r.install and args.names]:
            parser.error(f"no [install] in {', '.join(missing)}")
        images.prepare([r for r in recipes if r.install])
        return 0
    if args.command == "run":
        limits.enter()
        names = args.harness or recipe.available()
        recipes = [recipe.load(name) for name in names]
        scenarios = [s for selector in args.check for s in find(selector)] if args.check else all_scenarios()
        if not scenarios:
            parser.error("no check matches --check")
        overrides = {}
        for item in args.overrides:
            name, _, words = item.partition("=")
            overrides[name] = shlex.split(words)
        output = args.output or Path("results") / time.strftime("%Y-%m-%d-%H%M%S")
        for number in (signal.SIGTERM, signal.SIGHUP):
            signal.signal(number, interrupt)  # Stop as on Ctrl-C, leaving no agent behind.
        rows = runner.run(
            output,
            recipes,
            list(dict.fromkeys(scenarios)),
            args.api,
            args.tools,
            args.host,
            overrides,
            args.jobs,
            args.frontend or (["headless"] if args.quick else None),
            args.quick,
        )
        if args.quick:
            print(report.render([r for r in rows if r["level"] == REQUIRED]))
            print("\n" + report.fix_first(rows))
        else:
            print(report.render(rows))
        print(f"\nResults: {output}")
        return 1 if any(r["result"] in {"fail", "error"} for r in rows) else 0
    if args.command == "export":
        manifest, rows = runner.combine(args.runs)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(report.dumps(report.export(manifest, rows)))
        print(f"Wrote {args.output} ({len(rows)} rows)")
        return 0
    _, rows = runner.load(args.results)
    print(report.render(rows))
    if args.details:
        print("\n" + report.details(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main())
