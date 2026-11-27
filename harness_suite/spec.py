"""The tree of SPEC.md: areas, their sections, and the cases in each, with
the check ID of each case the suite measures. Each area names the frontend
its subject needs, so the area of a check decides where it runs."""

import functools
import re
from pathlib import Path

PATH = Path(__file__).resolve().parent.parent / "SPEC.md"
AREA = re.compile(r"^## (\d+)\. (.+)$")
SECTION = re.compile(r"^### (\d+\.\d+) (.+)$")
ITEM = re.compile(r"^- \[([ x-])\] (.*)$")
CHECK = re.compile(r"^`([a-z0-9.-]+)` ")
STATUS = {"x": "done", " ": "planned", "-": "out of scope"}
FRONTEND = re.compile(r"^Frontend: (.+)\.$")
FRONTENDS = {"headless": ("headless",), "TUI": ("tui",), "headless and TUI": ("headless", "tui")}


def tree(path: Path = PATH) -> list[dict]:
    """[{number, title, intro, frontends, sections: [{number, title, cases: [...]}]}]. A
    case is {status, check} when measured, else {status, text}. Cases before an
    area's first section go in a section with no title."""
    areas: list[dict] = []
    item: dict | None = None
    for line in path.read_text().splitlines():
        if found := AREA.match(line):
            areas.append({"number": found[1], "title": found[2], "intro": "", "frontends": None, "sections": []})
            item = None
        elif line.startswith("## "):
            break  # The notes after the numbered areas.
        elif not areas:
            continue
        elif found := FRONTEND.match(line):
            areas[-1]["frontends"] = FRONTENDS[found[1]]
        elif found := SECTION.match(line):
            areas[-1]["sections"].append({"number": found[1], "title": found[2], "cases": []})
            item = None
        elif found := ITEM.match(line):
            sections = areas[-1]["sections"]
            if not sections:
                sections.append({"number": areas[-1]["number"], "title": "", "cases": []})
            item = {"status": STATUS[found[1]], "text": found[2]}
            sections[-1]["cases"].append(item)
        elif line.startswith("  ") and item is not None:
            item["text"] += " " + line.strip()
        elif line.strip() and not areas[-1]["sections"]:
            areas[-1]["intro"] = (areas[-1]["intro"] + " " + line.strip()).strip()
        else:
            item = None
    for area in areas:
        if area["frontends"] is None:
            raise ValueError(f"SPEC.md: area {area['number']} names no frontend")
        for section in area["sections"]:
            for case in section["cases"]:
                if case["status"] == "done" and (found := CHECK.match(case["text"])):
                    case["check"] = found[1]
                    del case["text"]
    return areas


def order(path: Path = PATH) -> dict[str, int]:
    """Each measured check's place in the spec."""
    found = [c["check"] for a in tree(path) for s in a["sections"] for c in s["cases"] if "check" in c]
    return {check: number for number, check in enumerate(found)}


@functools.cache
def frontends() -> dict[str, tuple[str, ...]]:
    """The frontends each measured check runs on: those of its area."""
    return {c["check"]: a["frontends"] for a in tree() for s in a["sections"] for c in s["cases"] if "check" in c}
