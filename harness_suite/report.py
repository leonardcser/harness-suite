"""Read a results folder: a terminal grid, failure details, and a JSON export."""

import json
from collections import defaultdict

from . import recipe, spec
from .results import INFO, OPTIONAL, RECOMMENDED, REQUIRED
from .scenarios import all_checks

SECTIONS = (
    (REQUIRED, "Required: pass or fail"),
    (RECOMMENDED, "Recommended: pass or gap"),
    (OPTIONAL, "Optional: pass, fail, or unsupported"),
    (INFO, "Info: what the agent does"),
)
SHORT = {
    "chat": "chat",
    "responses": "resp",
    "anthropic": "anth",
    "mcp": "mcp",
    "native": "nat",
    "none": "",
    "headless": "",
    "tui": "tui",
}
# The worst result of a cell decides how it counts.
ORDER = ("error", "fail", "gap", "unsupported", "pass", "label", "skip")


def where(row: dict) -> str:
    parts = (SHORT[row["api"]], SHORT[row["tools"]], SHORT[row["frontend"]])
    return "/".join(part for part in parts if part)


def cell(rows: list[dict]) -> str:
    """One value if every case agrees; else each value and where."""
    measured = [r for r in rows if r["result"] != "skip"] or rows
    groups: dict[str, list[str]] = defaultdict(list)
    for row in measured:
        groups[row["summary"]].append(where(row) or row.get("case", ""))
    if len(groups) == 1:
        return next(iter(groups))
    return "; ".join(f"{summary} ({','.join(places)})" for summary, places in groups.items())


def grid(rows: list[dict]) -> dict:
    """{(check, variant): {agent: [rows]}}"""
    table: dict = defaultdict(lambda: defaultdict(list))
    for row in rows:
        table[(row["check"], row["variant"])][row["harness"]].append(row)
    return table


def render(rows: list[dict], width: int = 24) -> str:
    agents = sorted({r["harness"] for r in rows})
    table = grid(rows)
    order = spec.order()
    lines = []
    head = f"{'':<38}" + "".join(f"{h[: width - 1]:<{width}}" for h in agents)
    for level, title in SECTIONS:
        keys = sorted((k for k in table if level == first(table[k])["level"]), key=lambda k: (order.get(k[0], 999), k))
        if not keys:
            continue
        lines += ["", title, head]
        for check, variant in keys:
            name = check if variant == "default" else f"{check}/{variant}"
            cells = [cell(table[(check, variant)][h]) if table[(check, variant)][h] else "-" for h in agents]
            lines.append(f"{name[:37]:<38}" + "".join(f"{c[: width - 1]:<{width}}" for c in cells))
    if differ := frontends(rows):
        lines += ["", differ]
    lines += ["", totals(rows, agents)]
    return "\n".join(lines)


def frontends(rows: list[dict]) -> str:
    """Where an agent's TUI and headless runs of the same case disagree: the
    cases where testing one frontend alone would mislead."""
    pairs: dict = defaultdict(dict)
    for row in rows:
        if row["result"] != "skip":
            pairs[(row["harness"], row["check"], row["variant"], row["api"], row["tools"])][row["frontend"]] = row
    found = []
    for (agent, check, variant, api, tools), by in sorted(pairs.items()):
        headless, tui = by.get("headless"), by.get("tui")
        if headless and tui and (headless["result"], headless["summary"]) != (tui["result"], tui["summary"]):
            name = check if variant == "default" else f"{check}/{variant}"
            place = "/".join(p for p in (SHORT[api], SHORT[tools]) if p)
            said = [" ".join(row["summary"].split())[:60] for row in (headless, tui)]
            found.append(f"  {agent:<14} {name} [{place}]: headless {said[0]}; tui {said[1]}")
    return "\n".join(["Headless and TUI differ:", *found]) if found else ""


def first(per: dict) -> dict:
    return next(row for rows in per.values() for row in rows)


def totals(rows: list[dict], agents: list[str]) -> str:
    lines = ["Totals per agent (check variants; a variant counts by its worst case):"]
    table = grid(rows)
    for agent in agents:
        counts: dict = defaultdict(int)
        for per in table.values():
            results = {r["result"] for r in per.get(agent, [])}
            worst = next((r for r in ORDER if r in results), None)
            if worst:
                counts[worst] += 1
        lines.append(
            f"  {agent:<14} {counts['fail']:>3} fail, {counts['gap']:>3} gap, {counts['pass']:>3} pass,"
            f" {counts['unsupported']:>3} unsupported, {counts['label']:>3} info, {counts['skip']:>3} not measured,"
            f" {counts['error']:>3} suite errors"
        )
    return "\n".join(lines)


def details(rows: list[dict], results=("fail", "error", "gap")) -> str:
    lines = []
    for row in sorted(rows, key=lambda r: (r["harness"], r["check"], r["variant"], r["api"])):
        if row["result"] in results:
            lines.append(
                f"{row['harness']} {row['check']}/{row['variant']} [{where(row)}] {row['result']}: {row['summary']}"
            )
            if row["detail"]:
                lines.append(f"    {row['detail']}")
            if row.get("artifact"):
                lines.append(f"    {row['artifact']}")
    return "\n".join(lines)


def fix_first(rows: list[dict]) -> str:
    """The required checks that fail, in the spec's order: what a new agent
    should fix first, each with its evidence."""
    titles, order = {info.id: info.title for info, _ in all_checks()}, spec.order()
    failed = [r for r in rows if r["level"] == REQUIRED and r["result"] == "fail"]
    if not failed:
        return "Every required check passes."
    lines = ["Fix these first:"]
    ordered = sorted(failed, key=lambda r: (order.get(r["check"], len(order)), r["variant"], r["harness"]))
    for number, row in enumerate(ordered, 1):
        name = row["check"] if row["variant"] == "default" else f"{row['check']}/{row['variant']}"
        place = f" [{where(row)}]" if where(row) else ""
        lines.append(f"{number:>3}. {row['harness']} {name}{place}: {row['summary']}")
        if row["check"] in titles:
            lines.append(f"     {titles[row['check']]}")
        if row["detail"]:
            lines.append(f"     {row['detail']}")
        if row.get("artifact"):
            lines.append(f"     {row['artifact']}")
    return "\n".join(lines)


def about(name: str) -> dict:
    """What the recipe says of the agent, or its name alone once the recipe
    is gone."""
    try:
        return recipe.load(name).about
    except FileNotFoundError:
        return {"title": name, "repo": None, "tag_prefix": "v"}


def export(manifest: dict, rows: list[dict]) -> dict:
    """The checks, the agents, and every row, for other tools to read."""
    used = {r["check"] for r in rows}
    checks = []
    for info, scenario in all_checks():
        if info.id not in used:
            continue
        variants = [v for v in scenario.variants if info.applies(v)] if scenario.id else ["default"]
        checks.append(
            {
                "id": info.id,
                "title": info.title,
                "level": info.level,
                "description": info.description,
                "variants": variants,
            }
        )
    keys = ("harness", "api", "tools", "frontend", "case", "check", "variant", "result", "summary", "detail")
    return {
        "run": {k: manifest.get(k) for k in ("started", "finished", "commit", "source_sha256")},
        "harnesses": {
            k: {"version": v["version"], "runs_in": v.get("runs_in"), **about(v.get("name", k))}
            for k, v in manifest["harnesses"].items()
        },
        "checks": checks,
        "spec": spec.tree(),
        "rows": [{k: r[k] for k in keys} for r in rows],
    }


def dumps(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
