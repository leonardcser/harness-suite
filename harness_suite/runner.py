"""Run many cases into one results folder."""

import functools
import hashlib
import json
import subprocess
import threading
import time
from pathlib import Path

from . import engine, limits, live
from .recipe import ROOT, Recipe, available
from .recipe import load as load_recipe
from .results import REQUIRED
from .sandbox import Container, Host, image_exists
from .scenario import Scenario
from .scenarios import all_scenarios


def plan(
    recipes: list[Recipe],
    scenarios: list[type[Scenario]],
    apis: list[str] | None = None,
    sources: list[str] | None = None,
    frontends: list[str] | None = None,
) -> list[Scenario]:
    """Every case to run, as a scenario of one variant, agent, API, tool
    source and frontend: a case that does not `fit` is not one."""
    cases = []
    for recipe in recipes:
        for scenario in scenarios:
            chosen = [a for a in recipe.apis if not apis or a in apis]
            if scenario.per_harness:
                chosen = chosen[:1]
            for frontend in [f for f in scenario.frontends if not frontends or f in frontends]:
                for api in chosen:
                    for variant in scenario.variants:
                        tools = (
                            [s for s in scenario.sources if not sources or s in sources]
                            if scenario.uses_tools(variant)
                            else ["none"]
                        )
                        for source in tools:
                            case = scenario(variant, api, source, recipe, frontend)
                            if case.fits():
                                cases.append(case)
    return cases


def quick_cases(cases: list[Scenario]) -> list[Scenario]:
    """The cases that judge a required check, each variant once: on the first
    API and frontend, and with the agent's own tools if it has them, as a new
    agent has them before MCP."""
    kept, seen = [], set()
    for case in sorted(cases, key=lambda case: not (case.source == "native" and case.recipe.tools("native"))):
        key = (case.recipe.name, type(case), case.variant)
        if key in seen or not any(
            info.level == REQUIRED and info.applies(case.variant, case.api) for info, _ in case.checks()
        ):
            continue
        seen.add(key)
        kept.append(case)
    return kept


def commit() -> str | None:
    try:
        clean = not subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain", "harness_suite", "harnesses"], text=True
        )
        head = subprocess.check_output(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        return head if clean else head + "-dirty"
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def source_hash() -> str:
    digest = hashlib.sha256()
    for path in sorted((ROOT / "harness_suite").rglob("*.py")) + sorted((ROOT / "harnesses").glob("*.toml")):
        digest.update(str(path.relative_to(ROOT)).encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def sandboxes(recipes: list[Recipe], host: bool, commands: dict) -> dict:
    """Per agent: (sandbox factory or the reason it cannot run, where it
    runs). A recipe without an image runs on the host, from PATH."""
    found = {}
    for recipe in recipes:
        if host or recipe.name in commands or not recipe.image:
            command = commands.get(recipe.name, [])
            found[recipe.name] = (lambda command=command: Host(command)), "host"
        elif image_exists(recipe.image):
            found[recipe.name] = (lambda image=recipe.image: Container(image)), recipe.image
        else:
            reason = (
                f"No image {recipe.image}: build it with `python -m harness_suite images {recipe.name}`,"
                " or run the harness from PATH with --host"
            )
            found[recipe.name] = reason, None
    return found


def run(
    output: Path,
    recipes: list[Recipe],
    scenarios: list[type[Scenario]],
    apis=None,
    sources=None,
    host=False,
    commands=None,
    jobs=4,
    frontends=None,
    quick=False,
) -> list[dict]:
    """`quick`: only the cases of `quick_cases`."""
    output.mkdir(parents=True, exist_ok=False)
    factories = sandboxes(recipes, host, commands or {})
    # The longest cases first, so that none runs alone at the end.
    cases = plan(recipes, scenarios, apis, sources, frontends)
    cases = sorted(quick_cases(cases) if quick else cases, key=lambda case: -case.timeout)
    manifest = {
        "started": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "commit": commit(),
        "source_sha256": source_hash(),
        "harnesses": {
            r.name: {
                "name": r.name,
                "version": r.version,
                "image": r.image,
                "runs_in": factories[r.name][1],
                "command": (commands or {}).get(r.name),
                "recipe_sha256": hashlib.sha256(r.path.read_bytes()).hexdigest(),
            }
            for r in recipes
        },
        "cases": len(cases),
        "quick": quick,
    }
    (output / "run.json").write_text(json.dumps(manifest, indent=2))
    lock, rows, done = threading.Lock(), [], [0]
    log = (output / "results.jsonl").open("a")

    def one(case: Scenario) -> None:
        where = (case.recipe.name, case.frontend, case.api, case.source, case.id, case.variant)
        folder = output / "cases" / Path(*where)
        found = engine.case(case, factories[case.recipe.name][0], folder)
        for row in found:
            if "artifact" in row:
                row["artifact"] = str(Path(row["artifact"]).relative_to(output))
        with lock:
            rows.extend(found)
            done[0] += 1
            for row in found:
                log.write(json.dumps(row) + "\n")
            log.flush()
            print(line(case, found, done[0], len(cases)), flush=True)

    # At most `jobs` cases at once, each started only while the machine has room.
    slots, failed = threading.BoundedSemaphore(jobs), []

    def started(case: Scenario) -> None:
        try:
            one(case)
        except BaseException as error:
            failed.append(error)
        finally:
            slots.release()

    threads = []
    try:
        for case in cases:
            slots.acquire()
            if failed:
                break
            limits.wait_for_room()
            threads.append(threading.Thread(target=started, args=(case,), daemon=True))
            threads[-1].start()
        for thread in threads:
            thread.join()
        if failed:
            raise failed[0]
    except BaseException:
        # Ctrl-C or a suite bug: start nothing new, and stop every agent.
        live.stop_all()
        raise
    finally:
        for thread in threads:
            thread.join()
        log.close()
    manifest["finished"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (output / "run.json").write_text(json.dumps(manifest, indent=2))
    return gated(rows)


BASELINE = "tools.round-trip/text"  # The plain tool call that every other tool check builds on.


def gated(rows: list[dict]) -> list[dict]:
    """The rows, where a tool check's fail or gap becomes not measured if the
    plain tool round trip of its harness, frontend, API and tool source did
    not pass: the check would only repeat that failure."""
    group = lambda r: (r["harness"], r["frontend"], r["api"], r["tools"])  # noqa: E731
    baseline = {group(r): r for r in rows if r["case"] == BASELINE and r["check"] == "tools.round-trip"}
    out = []
    for r in rows:
        base = baseline.get(group(r))
        if r.get("uses_tools") and r["case"] != BASELINE and base and base["result"] != "pass":
            if r["result"] in ("fail", "gap"):
                why = f"The plain tool round trip gave {base['result']} ({base['summary']}); this check gave {r['summary']}."
                r = {**r, "result": "skip", "summary": "not measured", "detail": f"{why} {r['detail']}".strip()}
        out.append(r)
    return out


def line(scenario: Scenario, rows: list[dict], done: int, total: int) -> str:
    where = f"{scenario.recipe.name}/{scenario.frontend}/{scenario.api}/{scenario.source}"
    shown = []
    for row in rows:
        if row["check"] == engine.VALID.id and row["result"] == "pass":
            continue
        name = "" if row["check"] == scenario.id else f"{row['check']}: "
        detail = f" ({row['detail'][:80]})" if row["result"] in {"fail", "error"} and row["detail"] else ""
        shown.append(f"{name}{row['summary']}{detail}")
    return f"[{done:>4}/{total}] {where:<36} {scenario.id}/{scenario.variant:<24} {' | '.join(shown)}"


def load(directory: Path) -> tuple[dict, list[dict]]:
    """A run's manifest and rows, gated on the tool baseline."""
    manifest, rows = raw(directory)
    return manifest, gated(rows)


def raw(directory: Path) -> tuple[dict, list[dict]]:
    manifest = json.loads((directory / "run.json").read_text())
    rows = [json.loads(line) for line in (directory / "results.jsonl").read_text().splitlines() if line]
    return manifest, rows


def combine(directories: list[Path]) -> tuple[dict, list[dict]]:
    """Runs as one: each case comes from the last run that has it. Rows of
    cases, checks or variants the suite no longer has are left out, so a
    rerun of the scenarios that changed makes an older full run current."""
    cases: dict[tuple, list[dict]] = {}
    manifest: dict = {}
    for directory in directories:
        found, rows = raw(directory)
        harnesses = {**manifest.get("harnesses", {}), **found["harnesses"]}
        manifest = {**found, "started": manifest.get("started", found["started"]), "harnesses": harnesses}
        fresh: dict[tuple, list[dict]] = {}
        for row in rows:
            key = (row["harness"], row["frontend"], row["api"], row["tools"], row["case"])
            fresh.setdefault(key, []).append(row)
        cases |= fresh
    rows = [r for group in cases.values() for r in group]
    return manifest, gated([r for r in rows if current(r)])


@functools.cache
def recipe_named(name: str) -> Recipe | None:
    return load_recipe(name) if name in available() else None


def current(row: dict) -> bool:
    """Whether the suite still has the row's case and check. For an agent
    without a recipe here, the check alone decides."""
    scenario_id, variant = row["case"].rsplit("/", 1)
    scenario = next((s for s in all_scenarios() if s.id == scenario_id), None)
    if scenario is None or variant not in scenario.variants:
        return False
    if row["check"] == engine.VALID.id:
        info = engine.VALID
    else:
        info = next((info for info, _ in scenario.checks() if info.id == row["check"]), None)
        if info is None or not info.applies(variant, row["api"]):
            return False
    if (recipe := recipe_named(row["harness"])) is None:
        return True
    return scenario(variant, row["api"], row["tools"], recipe, row["frontend"]).fits()
