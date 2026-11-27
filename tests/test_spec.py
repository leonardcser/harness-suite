"""SPEC.md and the suite's checks list the same cases."""

import re
import unittest
from pathlib import Path

from harness_suite import spec
from harness_suite.scenarios import all_checks, all_scenarios

SPEC = Path(__file__).parent.parent / "SPEC.md"


class Spec(unittest.TestCase):
    def test_every_check_is_in_the_spec_once(self):
        listed = re.findall(r"^- \[x\] `([a-z0-9.-]+)`", SPEC.read_text(), re.MULTILINE)
        checks = [info.id for info, _ in all_checks()]
        self.assertEqual(sorted(listed), sorted(set(listed)), "a check is listed twice")
        self.assertEqual(sorted(set(checks) - set(listed)), [], "checks missing from SPEC.md")
        self.assertEqual(sorted(set(listed) - set(checks)), [], "SPEC.md lists checks that do not exist")

    def test_tree_holds_every_check(self):
        found = [c["check"] for a in spec.tree() for s in a["sections"] for c in s["cases"] if "check" in c]
        self.assertEqual(sorted(found), sorted(info.id for info, _ in all_checks()))

    def test_a_scenarios_checks_share_one_frontend(self):
        for scenario in all_scenarios():
            with self.subTest(scenario.id):
                found = {spec.frontends()[info.id] for info, _ in scenario.checks()}
                self.assertEqual(found, {scenario.frontends})


if __name__ == "__main__":
    unittest.main()
