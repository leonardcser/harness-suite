"""The fake provider accepts what the real APIs accept and rejects the rest."""

import base64
import unittest

from harness_suite.apis import anthropic, chat, responses
from harness_suite.apis.common import picture
from harness_suite.fixture import PICTURES, RED, bmp, png
from harness_suite.messages import Call, Reply, opaque, signature

APIS = {"chat": chat, "responses": responses, "anthropic": anthropic}
TOOL = {"type": "function", "function": {"name": "shell", "parameters": {"type": "object"}}}
CALL = {"id": "c1", "type": "function", "function": {"name": "shell", "arguments": "{}"}}
USE = {"type": "tool_use", "id": "t1", "name": "shell", "input": {}}
RESULT = {"type": "tool_result", "tool_use_id": "t1", "content": "ok"}
THOUGHT = {"type": "thinking", "thinking": "plan", "signature": signature("plan")}
REDACTED = {"type": "redacted_thinking", "data": opaque("secret")}
OMITTED = {"type": "thinking", "thinking": "", "signature": opaque("hidden")}
THINKING_ON = {"thinking": {"type": "enabled", "budget_tokens": 1024}}


def tool_turn(*thinking):
    """A request whose open tool turn starts with these thinking blocks."""
    messages = ask(
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [*thinking, USE]},
        {"role": "user", "content": [RESULT]},
    )
    return {**messages, **THINKING_ON}


def issued(*blocks):
    return {"turns": {"t1": [anthropic.key(b) for b in blocks]}, "opaque": {opaque("secret"), opaque("hidden")}}


CHAT_TURN = [
    {"role": "user", "content": "hi"},
    {"role": "assistant", "tool_calls": [CALL], "reasoning_content": "plan"},
    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
]
RS = {"type": "reasoning", "id": "rs1", "summary": [], "encrypted_content": opaque("x")}
FC = {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"}
FO = {"type": "function_call_output", "call_id": "c1", "output": "ok"}


def ask(*messages):
    return {"model": "m", "max_tokens": 4096, "messages": list(messages)}


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def claude_image(data: bytes, media_type: str = "image/png") -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media_type, "data": b64(data)}}


def openai_image(data: bytes, media_type: str = "image/png") -> dict:
    return {"type": "image_url", "image_url": {"url": f"data:{media_type};base64,{b64(data)}"}}


def in_result(image: dict) -> dict:
    """A tool turn whose result holds the image."""
    return ask(
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [USE]},
        {"role": "user", "content": [{**RESULT, "content": [image]}]},
    )


WIDE = PICTURES["wide"][0]()
HEAVY = PICTURES["heavy"][0]()


# (api, body, store, a piece of the error, or None when the real API accepts it)
CASES = {
    "chat: a call and its result": (
        "chat",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": None, "tool_calls": [CALL]},
            {"role": "tool", "tool_call_id": "c1", "content": "ok"},
        ),
        {},
        None,
    ),
    "chat: a tool name with a dot": (
        "chat",
        {**ask({"role": "user", "content": "hi"}), "tools": [{"type": "function", "function": {"name": "a.b"}}]},
        {},
        "tools[0].function.name",
    ),
    "chat: a result with no call": (
        "chat",
        ask({"role": "user", "content": "hi"}, {"role": "tool", "tool_call_id": "c1", "content": "ok"}),
        {},
        "must be a response",
    ),
    "chat: a call with no result": (
        "chat",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "tool_calls": [CALL]},
            {"role": "user", "content": "next"},
        ),
        {},
        "did not have response messages: c1",
    ),
    "chat: a call left open at the end": (
        "chat",
        ask({"role": "user", "content": "hi"}, {"role": "assistant", "tool_calls": [CALL]}),
        {},
        "c1",
    ),
    "responses: a call and its output": (
        "responses",
        {
            "model": "m",
            "input": [
                {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
        },
        {},
        None,
    ),
    "responses: an output with no call": (
        "responses",
        {"model": "m", "input": [{"type": "function_call_output", "call_id": "c1", "output": "ok"}]},
        {},
        "No tool call found",
    ),
    "responses: a call with no output": (
        "responses",
        {"model": "m", "input": [{"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"}]},
        {},
        "No tool output found",
    ),
    "responses: a call without its reasoning item": (
        "responses",
        {
            "model": "m",
            "input": [
                {"type": "function_call", "id": "fc1", "call_id": "c1", "name": "shell", "arguments": "{}"},
                {"type": "function_call_output", "call_id": "c1", "output": "ok"},
            ],
        },
        {"requires": {"fc1": "rs1"}},
        "without its required 'reasoning' item",
    ),
    "anthropic: a use and its result": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [RESULT]},
        ),
        {},
        None,
    ),
    "anthropic: parallel results in separate user messages": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE, {**USE, "id": "t2"}]},
            {"role": "user", "content": [RESULT]},
            {"role": "user", "content": [{**RESULT, "tool_use_id": "t2"}]},
        ),
        {},
        None,
    ),
    "anthropic: an empty text block": (
        "anthropic",
        ask({"role": "user", "content": [{"type": "text", "text": ""}]}),
        {},
        "must be non-empty",
    ),
    "anthropic: an empty system block": (
        "anthropic",
        {**ask({"role": "user", "content": "hi"}), "system": [{"type": "text", "text": ""}]},
        {},
        "system: text content",
    ),
    "anthropic: a tool ID with a dot": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{**USE, "id": "a.b"}]},
            {"role": "user", "content": [{**RESULT, "tool_use_id": "a.b"}]},
        ),
        {},
        "tool_use.id",
    ),
    "anthropic: the same tool ID twice": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [RESULT]},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [RESULT]},
        ),
        {},
        "ids must be unique",
    ),
    "anthropic: text before a result": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [{"type": "text", "text": "x"}, RESULT]},
        ),
        {},
        "must come first",
    ),
    "anthropic: a result for another call": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [RESULT, {**RESULT, "tool_use_id": "t9"}]},
        ),
        {},
        "unexpected `tool_use_id`",
    ),
    "anthropic: a use with no result": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": "next"},
        ),
        {},
        "without `tool_result`",
    ),
    "anthropic: thinking with a changed signature": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "plan", "signature": signature("other")},
                    {"type": "text", "text": "x"},
                ],
            },
        ),
        {},
        "Invalid `signature`",
    ),
    "anthropic: thinking without a signature": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "thinking", "thinking": "plan"}, {"type": "text", "text": "x"}]},
        ),
        {},
        "signature: Field required",
    ),
    "anthropic: thinking dropped from the open tool turn": (
        "anthropic",
        {
            **ask(
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [USE]},
                {"role": "user", "content": [RESULT]},
            ),
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
        {"turns": {"t1": [("thinking", signature("plan"))]}},
        "must start with a thinking block",
    ),
    "anthropic: adaptive thinking dropped from the open tool turn": (
        "anthropic",
        {
            **ask(
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": [USE]},
                {"role": "user", "content": [RESULT]},
            ),
            "thinking": {"type": "adaptive"},
        },
        {"turns": {"t1": [("thinking", signature("plan"))]}},
        None,
    ),
    "anthropic: thinking kept on the open tool turn": (
        "anthropic",
        {
            **ask(
                {"role": "user", "content": "hi"},
                {
                    "role": "assistant",
                    "content": [{"type": "thinking", "thinking": "plan", "signature": signature("plan")}, USE],
                },
                {"role": "user", "content": [RESULT]},
            ),
            "thinking": {"type": "enabled", "budget_tokens": 1024},
        },
        {"turns": {"t1": [("thinking", signature("plan"))]}},
        None,
    ),
    "anthropic: four cache marks": (
        "anthropic",
        {
            **ask(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": t, "cache_control": {"type": "ephemeral"}} for t in "abc"],
                }
            ),
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        },
        {},
        None,
    ),
    "anthropic: five cache marks": (
        "anthropic",
        {
            **ask(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": t, "cache_control": {"type": "ephemeral"}} for t in "abcd"],
                }
            ),
            "cache_control": {"type": "ephemeral"},
        },
        {},
        "A maximum of 4 blocks with cache_control",
    ),
    "anthropic: redacted thinking kept": ("anthropic", tool_turn(THOUGHT, REDACTED), issued(THOUGHT, REDACTED), None),
    "anthropic: redacted thinking dropped": (
        "anthropic",
        tool_turn(THOUGHT),
        issued(THOUGHT, REDACTED),
        "cannot be modified",
    ),
    "anthropic: thinking blocks reordered": (
        "anthropic",
        tool_turn(REDACTED, THOUGHT),
        issued(THOUGHT, REDACTED),
        "cannot be modified",
    ),
    "anthropic: redacted data changed": (
        "anthropic",
        tool_turn({**REDACTED, "data": "changed"}),
        issued(REDACTED),
        "cannot be modified",
    ),
    "anthropic: omitted thinking kept with empty text": ("anthropic", tool_turn(OMITTED), issued(OMITTED), None),
    "anthropic: omitted thinking with text filled in": (
        "anthropic",
        tool_turn({**OMITTED, "thinking": "guessed"}),
        issued(OMITTED),
        None,
    ),
    "chat: reasoning kept in the tool loop": (
        "chat",
        {"model": "m", "messages": CHAT_TURN},
        {"reasoning": {"c1": "plan"}},
        None,
    ),
    "chat: reasoning dropped in the tool loop": (
        "chat",
        {"model": "m", "messages": [CHAT_TURN[0], {**CHAT_TURN[1], "reasoning_content": None}, CHAT_TURN[2]]},
        {"reasoning": {"c1": "plan"}},
        "must be passed back",
    ),
    "chat: reasoning dropped after the tool loop": (
        "chat",
        {
            "model": "m",
            "messages": [
                CHAT_TURN[0],
                {**CHAT_TURN[1], "reasoning_content": None},
                CHAT_TURN[2],
                {"role": "assistant", "content": "done"},
                {"role": "user", "content": "next"},
            ],
        },
        {"reasoning": {"c1": "plan"}},
        None,
    ),
    "anthropic: an empty text block in a tool result": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [{**RESULT, "content": [{"type": "text", "text": ""}]}]},
        ),
        {},
        "text content blocks must be non-empty",
    ),
    "anthropic: a tool result with no content": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [{**RESULT, "content": []}]},
        ),
        {},
        None,
    ),
    "anthropic: two tools with one name": (
        "anthropic",
        {**ask({"role": "user", "content": "hi"}), "tools": [{"name": "t", "input_schema": {"type": "object"}}] * 2},
        {},
        "Tool names must be unique",
    ),
    "anthropic: a small PNG": ("anthropic", ask({"role": "user", "content": [claude_image(RED)]}), {}, None),
    "anthropic: a BMP": (
        "anthropic",
        ask({"role": "user", "content": [claude_image(bmp(1, 2, 3), "image/bmp")]}),
        {},
        "Input should be 'image/jpeg'",
    ),
    "anthropic: a BMP declared as PNG, in a tool result": (
        "anthropic",
        in_result(claude_image(bmp(1, 2, 3))),
        {},
        "does not match the provided media type image/png",
    ),
    "anthropic: an image over 8000 px": ("anthropic", in_result(claude_image(WIDE)), {}, "8000 pixels"),
    "anthropic: an image over 10 MB of base64": ("anthropic", in_result(claude_image(HEAVY)), {}, "exceeds 10 MB"),
    "anthropic: temperature 0.3 with thinking on": (
        "anthropic",
        {**ask({"role": "user", "content": "hi"}), **THINKING_ON, "temperature": 0.3},
        {},
        "may only be set to 1",
    ),
    "anthropic: temperature 0.3 with thinking off": (
        "anthropic",
        {**ask({"role": "user", "content": "hi"}), "temperature": 0.3},
        {},
        None,
    ),
    "anthropic: top_p 0.9 with thinking on": (
        "anthropic",
        {**ask({"role": "user", "content": "hi"}), **THINKING_ON, "top_p": 0.9},
        {},
        "greater than or equal to 0.95",
    ),
    "anthropic: a thinking budget over max_tokens": (
        "anthropic",
        {**ask({"role": "user", "content": "hi"}), "thinking": {"type": "enabled", "budget_tokens": 8192}},
        {},
        "must be greater than `thinking.budget_tokens`",
    ),
    "anthropic: a thinking budget under 1024": (
        "anthropic",
        {**ask({"role": "user", "content": "hi"}), "thinking": {"type": "enabled", "budget_tokens": 512}},
        {},
        "greater than or equal to 1024",
    ),
    "anthropic: a text block of whitespace beside a tool call": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{"type": "text", "text": "\n\n"}, USE]},
            {"role": "user", "content": [RESULT]},
        ),
        {},
        "must contain non-whitespace text",
    ),
    "anthropic: a tool result of whitespace": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [USE]},
            {"role": "user", "content": [{**RESULT, "content": "\n"}]},
        ),
        {},
        None,
    ),
    "anthropic: a tool name over 200 characters in the history": (
        "anthropic",
        ask(
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": [{**USE, "name": "x" * 201}]},
            {"role": "user", "content": [RESULT]},
        ),
        {},
        "at most 200 characters",
    ),
    "anthropic: a 1h cache mark after a 5m one": (
        "anthropic",
        {
            **ask(
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
                }
            ),
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral"}}],
        },
        {},
        "must not come after",
    ),
    "anthropic: a 1h cache mark before a 5m one": (
        "anthropic",
        {
            **ask(
                {"role": "user", "content": [{"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}}]}
            ),
            "system": [{"type": "text", "text": "s", "cache_control": {"type": "ephemeral", "ttl": "1h"}}],
        },
        {},
        None,
    ),
    "chat: an image over Anthropic's limits": (
        "chat",
        ask({"role": "user", "content": [openai_image(WIDE), openai_image(HEAVY)]}),
        {},
        None,
    ),
    "chat: a BMP": (
        "chat",
        ask({"role": "user", "content": [openai_image(bmp(1, 2, 3), "image/bmp")]}),
        {},
        "unsupported image",
    ),
    "responses: a BMP": (
        "responses",
        {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_image", "image_url": f"data:image/bmp;base64,{b64(bmp(1, 2, 3))}"}],
                }
            ],
        },
        {},
        "unsupported image",
    ),
    "responses: reasoning by its ID alone, not stored": (
        "responses",
        {"model": "m", "input": [{"type": "reasoning", "id": "rs1", "summary": []}, FC, FO]},
        {},
        "Items are not persisted",
    ),
    "responses: reasoning by its ID alone, stored": (
        "responses",
        {"model": "m", "input": [{"type": "reasoning", "id": "rs1", "summary": []}, FC, FO]},
        {"items": {"rs1": RS}},
        None,
    ),
    "responses: a previous response never stored": (
        "responses",
        {"model": "m", "previous_response_id": "resp_1", "input": "hi"},
        {},
        "Previous response with id 'resp_1' not found",
    ),
    "responses: encrypted reasoning kept": (
        "responses",
        {"model": "m", "input": [RS, FC, FO]},
        {"encrypted": {opaque("x")}},
        None,
    ),
    "responses: encrypted reasoning changed": (
        "responses",
        {"model": "m", "input": [{**RS, "encrypted_content": "changed"}, FC, FO]},
        {"encrypted": {opaque("x")}},
        "could not be verified",
    ),
}


class Validation(unittest.TestCase):
    def test_rules(self):
        for name, (api, body, store, expected) in CASES.items():
            with self.subTest(name):
                module = APIS[api]
                found = module.problems(module.parse(body, {}, store), store)
                if expected is None:
                    self.assertEqual(found, [])
                else:
                    self.assertIn(expected, " ".join(r.text for r in found))

    def test_interleaved_thinking_budget_may_exceed_max_tokens(self):
        body = {**ask({"role": "user", "content": "hi"}), "thinking": {"type": "enabled", "budget_tokens": 8192}}
        for betas, expected in (("", True), ("claude-code-20250219, interleaved-thinking-2025-05-14", False)):
            with self.subTest(betas=betas):
                headers = {"Anthropic-Beta": betas} if betas else {}
                found = anthropic.problems(anthropic.parse(body, headers, {}), {})
                self.assertEqual(bool(found), expected, found)


class Stateless(unittest.TestCase):
    def test_store_false_keeps_nothing_and_include_brings_the_reasoning(self):
        store: dict = {}
        reply = Reply(thinking="plan", calls=(Call("c1", "shell", "{}"),))
        for flags, encrypted in (({}, False), ({"include": ["reasoning.encrypted_content"]}, True)):
            body = {"model": "m", "input": "hi", "store": False, **flags}
            output = responses.body(responses.parse(body, {}, store), reply, 1, store)["output"]
            self.assertEqual("encrypted_content" in output[0], encrypted)
        self.assertEqual(store.get("responses", {}), {})
        self.assertEqual(store.get("items", {}), {})


class Pictures(unittest.TestCase):
    def test_format_and_size_from_the_bytes(self):
        gif = b"GIF89a" + (300).to_bytes(2, "little") + (200).to_bytes(2, "little") + b"\0" * 10
        jpeg = bytes.fromhex("ffd8ffe000104a46494600010100000100010000ffc0001108012c01f403012200021101031101")
        webp = b"RIFF\0\0\0\0WEBPVP8X" + b"\0" * 8 + (639).to_bytes(3, "little") + (479).to_bytes(3, "little")
        for data, expected in [
            (png(1, 2, 3, width=9000, height=16), ("png", 9000, 16)),
            (gif, ("gif", 300, 200)),
            (jpeg, ("jpeg", 500, 300)),
            (webp, ("webp", 640, 480)),
            (bmp(1, 2, 3), (None, 0, 0)),
        ]:
            with self.subTest(expected[0]):
                self.assertEqual(picture(data), expected)

    def test_heavy_picture_is_just_over_ten_mb_of_base64(self):
        self.assertGreater(len(base64.b64encode(HEAVY)), 10 * 1024**2)
        self.assertLess(len(HEAVY), 10 * 1024**2)


class Parsing(unittest.TestCase):
    def test_all_apis_agree_on_a_tool_turn(self):
        bodies = {
            "chat": {
                **ask(
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "tool_calls": [CALL]},
                    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                ),
                "tools": [TOOL],
            },
            "responses": {
                "model": "m",
                "input": [
                    {"role": "user", "content": "hi"},
                    {"type": "function_call", "call_id": "c1", "name": "shell", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "c1", "output": "ok"},
                ],
                "tools": [{"type": "function", "name": "shell", "parameters": {"type": "object"}}],
            },
            "anthropic": {
                **ask(
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": [{**USE, "id": "c1"}]},
                    {"role": "user", "content": [{**RESULT, "tool_use_id": "c1"}]},
                ),
                "tools": [{"name": "shell", "input_schema": {"type": "object"}}],
            },
        }
        for api, body in bodies.items():
            with self.subTest(api):
                request = APIS[api].parse(body, {}, {})
                self.assertEqual(request.model, "m")
                self.assertEqual([t.name for t in request.tools], ["shell"])
                self.assertEqual([(c.id, c.name, c.args()) for c in request.calls], [("c1", "shell", {})])
                self.assertEqual(request.result("c1").text, "ok")
                self.assertEqual(request.user_texts, ["hi"])


if __name__ == "__main__":
    unittest.main()
