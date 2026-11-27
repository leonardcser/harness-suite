"""The fake model tells the harness's task turns from its side requests."""

import unittest

from harness_suite.apis import chat
from harness_suite.evidence import Run, Served
from harness_suite.messages import Call
from harness_suite.model import Model
from harness_suite.scenarios import find
from harness_suite.script import TASK
from tests.test_checks import fake

TOOLS = [{"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}]
PROMPT = {"role": "user", "content": f"{TASK}: do it"}


def turn(call_id: str) -> list[dict]:
    call = {"id": call_id, "type": "function", "function": {"name": "shell", "arguments": "{}"}}
    return [{"role": "assistant", "tool_calls": [call]}, {"role": "tool", "tool_call_id": call_id, "content": "ok"}]


class Classify(unittest.TestCase):
    def setUp(self):
        scenario = find("tools.sequence")[0]("default", "chat", "native", fake())
        self.model = Model(scenario, Run("default", "chat", "native", "headless"), {})
        self.model.issued = {"c1", "c2"}
        self.model.tracks[0].pending = Served(1, 2, [Call("c2", "shell", "{}")], [])
        self.model.tracks[0].cut = "SUITE_PARTIAL " * 10  # A reply whose stream broke off.

    def task(self, *messages, tools=TOOLS) -> bool:
        body = {"model": "suite-model", "messages": [PROMPT, *messages], "tools": tools}
        return self.model.classify(chat.parse(body, {}, {}), self.model.tracks[0])[0]

    def test_requests(self):
        cases = {
            "the user's prompt": (True, ()),
            "results the model waits for": (True, (*turn("c1"), *turn("c2"))),
            "those results with a reminder": (True, (*turn("c2"), {"role": "user", "content": "<reminder>"})),
            "older results alone, as a retry": (True, turn("c1")),
            "older results with an instruction": (False, (*turn("c1"), {"role": "user", "content": "Describe it."})),
            "a summary request after results": (
                False,
                (*turn("c2"), {"role": "user", "content": "Stop and produce a handoff summary."}),
            ),
            "a checkpoint with our summary": (
                True,
                (
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "Summary so far: SUITE_AUX_REPLY"},
                ),
            ),
            "a checkpoint that says what was summarized": (
                True,
                (
                    {"role": "assistant", "content": "ok"},
                    {"role": "user", "content": "Earlier work, summarized: SUITE_AUX_REPLY"},
                ),
            ),
            "a title turn on the whole conversation, tools off": (
                False,
                (
                    {"role": "user", "content": "Ephemeral side-channel turn. Do NOT emit tool calls."},
                    {"role": "user", "content": "<title-request>\nName this session.\n</title-request>"},
                ),
            ),
            "plain text after the task": (
                False,
                ({"role": "assistant", "content": "ok"}, {"role": "user", "content": "Name this chat."}),
            ),
            "results the model never sent": (False, turn("c9")),
            "a request to go on after the reply broke off": (
                True,
                (
                    {"role": "assistant", "content": "SUITE_PARTIAL SUITE_PARTIAL"},
                    {"role": "user", "content": "The reply broke off. Go on. Do not repeat or summarize it."},
                ),
            ),
        }
        for name, (expected, messages) in cases.items():
            with self.subTest(name):
                self.assertEqual(self.task(*messages), expected)

    def test_no_tools_is_a_side_request(self):
        self.assertFalse(self.task(tools=[]))


if __name__ == "__main__":
    unittest.main()
