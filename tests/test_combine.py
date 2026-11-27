"""Runs combine into one result set for export."""

import json
import tempfile
import unittest
from pathlib import Path

from harness_suite import runner


def row(case: str, check: str, variant: str, result: str) -> dict:
    where = {"harness": "fake", "frontend": "headless", "api": "chat", "tools": "none"}
    return {**where, "case": case, "check": check, "variant": variant, "result": result}


def write(folder: Path, started: str, rows: list[dict], harnesses=("fake",)) -> Path:
    folder.mkdir()
    agents = {name: {"version": "1"} for name in harnesses}
    (folder / "run.json").write_text(json.dumps({"started": started, "harnesses": agents}))
    (folder / "results.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    return folder


class Combine(unittest.TestCase):
    def test_later_runs_replace_cases_and_old_checks_drop(self):
        with tempfile.TemporaryDirectory() as root:
            full = write(
                Path(root, "full"),
                "first",
                [
                    row("request.model/plain", "request.model", "plain", "fail"),
                    row("request.model/plain", "request.valid", "default", "pass"),
                    row("request.model/dots", "request.model", "dots", "pass"),
                    row("cli.sigterm/default", "cli.sigterm", "default", "pass"),
                ],
                harnesses=("fake", "other"),
            )
            rerun = write(Path(root, "rerun"), "second", [row("request.model/plain", "request.model", "plain", "pass")])
            manifest, rows = runner.combine([full, rerun])
        found = sorted((r["case"], r["check"], r["result"]) for r in rows)
        self.assertEqual(
            found,
            [
                ("request.model/dots", "request.model", "pass"),
                ("request.model/plain", "request.model", "pass"),
            ],
        )
        self.assertEqual(manifest["started"], "first")
        self.assertEqual(sorted(manifest["harnesses"]), ["fake", "other"])


def tool_row(case: str, check: str, result: str, uses_tools=True, api="chat") -> dict:
    variant = case.rsplit("/", 1)[1]
    found = {**row(case, check, variant, result), "tools": "mcp", "api": api, "uses_tools": uses_tools}
    return {**found, "summary": result, "detail": ""}


class Gated(unittest.TestCase):
    def test_a_failed_baseline_leaves_the_groups_tool_findings_unmeasured(self):
        rows = runner.gated(
            [
                tool_row("tools.round-trip/text", "tools.round-trip", "fail"),
                tool_row("tools.parallel/two", "tools.parallel", "fail"),
                tool_row("tools.unicode/emoji", "tools.unicode", "pass"),
                tool_row("request.model/plain", "request.model", "fail", uses_tools=False),
                tool_row("tools.parallel/two", "tools.parallel", "fail", api="responses"),
            ]
        )
        found = [(r["api"], r["case"], r["result"]) for r in rows]
        self.assertEqual(
            found,
            [
                ("chat", "tools.round-trip/text", "fail"),
                ("chat", "tools.parallel/two", "skip"),
                ("chat", "tools.unicode/emoji", "pass"),
                ("chat", "request.model/plain", "fail"),
                ("responses", "tools.parallel/two", "fail"),
            ],
        )
        self.assertIn("plain tool round trip gave fail", rows[1]["detail"])


if __name__ == "__main__":
    unittest.main()
