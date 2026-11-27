"""The fake agent's three wire APIs: Chat Completions, Responses and
Anthropic Messages.

The agent keeps its conversation as Chat messages. An assistant message may
also hold `reasoning_items`: the reasoning exactly as one API returned it
(Responses reasoning items, Anthropic thinking blocks), which goes back
unchanged to that API and is left out on another.
"""

import json
import sys

KEY = "suite-placeholder"
# Words of a provider error that says the request is over the context window.
OVERFLOW = (
    "context_length_exceeded",
    "prompt is too long",
    "exceeds the context window",
    "maximum context length",
    "exceed_context_size_error",
    "exceed context limit",
)


# Stop reasons that say a reply ended before it was complete.
EARLY = {"length", "error", "max_tokens", "model_context_window_exceeded", "incomplete"}


class Turn:
    """What one reply said."""

    def __init__(self):
        self.text, self.reasoning, self.done, self.stop = "", "", False, ""
        self.calls: dict[int, dict] = {}  # Position: {id, name, arguments}.
        self.items: list[dict] = []  # Reasoning as the API returned it.

    def call(self, position, id="", name="") -> dict:
        return self.calls.setdefault(position, {"id": id, "name": name, "arguments": ""})

    @property
    def early(self) -> bool:
        """The reply ended without its finish events, or with a stop reason
        that says it is not complete."""
        return not self.done or self.stop in EARLY


class StreamError(ConnectionError):
    """The provider ended the stream with an error event."""


# Requests.


def request(
    api: str, model: str, messages: list, tools: list, options: dict, marks: bool = True
) -> tuple[str, dict, dict]:
    """(path after the base URL, headers, body) of one streamed request. With
    `marks`, an Anthropic request asks the provider to cache its prompt."""
    if api == "responses":
        return "/responses", {"Authorization": f"Bearer {KEY}"}, responses_body(model, messages, tools, options)
    if api == "anthropic":
        headers = {"x-api-key": KEY, "anthropic-version": "2023-06-01"}
        return "/messages", headers, anthropic_body(model, messages, tools, options, marks)
    body = {"model": model, "stream": True, "messages": [chat_message(m) for m in messages], "tools": tools}
    return "/chat/completions", {"Authorization": f"Bearer {KEY}"}, {**body, **options}


def chat_message(message: dict) -> dict:
    return {k: v for k, v in message.items() if k != "reasoning_items"}


def own_items(message: dict, api: str) -> list[dict]:
    held = message.get("reasoning_items") or {}
    return held.get("items", []) if held.get("api") == api else []


def system_text(messages: list) -> str:
    return "\n\n".join(m["content"] for m in messages if m["role"] == "system")


def responses_body(model: str, messages: list, tools: list, options: dict) -> dict:
    items: list[dict] = []
    for message in messages:
        role, content = message["role"], message.get("content")
        if role == "user":
            items.append({"role": "user", "content": responses_parts(content)})
        elif role == "assistant":
            items += own_items(message, "responses")
            if content:
                items.append(
                    {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": content}]}
                )
            for call in message.get("tool_calls") or []:
                function = call["function"]
                items.append(
                    {
                        "type": "function_call",
                        "call_id": call["id"],
                        "name": function["name"],
                        "arguments": function["arguments"],
                    }
                )
        elif role == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"], "output": content})
    body = {
        "model": model,
        "stream": True,
        "store": False,
        "include": ["reasoning.encrypted_content"],
        "input": items,
        "tools": [{"type": "function", **t["function"]} for t in tools],
        **options,
    }
    if system := system_text(messages):
        body["instructions"] = system
    return body


def responses_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "input_text", "text": content}]
    return [
        {"type": "input_text", "text": p["text"]}
        if p["type"] == "text"
        else {"type": "input_image", "image_url": p["image_url"]["url"]}
        for p in content
    ]


def anthropic_body(model: str, messages: list, tools: list, options: dict, marks: bool) -> dict:
    turns: list[dict] = []

    def add(role: str, blocks: list[dict]) -> None:
        if not blocks:
            return
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += blocks
        else:
            turns.append({"role": role, "content": blocks})

    for message in messages:
        role, content = message["role"], message.get("content")
        if role == "user":
            add("user", anthropic_parts(content))
        elif role == "assistant":
            blocks = [dict(block) for block in own_items(message, "anthropic")]
            if content and (
                content.strip() or "--keep-blank-text" in sys.argv
            ):  # The API takes no text of whitespace alone.
                blocks.append({"type": "text", "text": content})
            for call in message.get("tool_calls") or []:
                function = call["function"]
                try:
                    arguments = json.loads(function["arguments"] or "{}")
                except ValueError:
                    arguments = {}
                # The API takes names of at most 200 characters, also in the history.
                name = function["name"] if "--keep-long-names" in sys.argv else function["name"][:200]
                blocks.append({"type": "tool_use", "id": call["id"], "name": name, "input": arguments})
            add("assistant", blocks)
        elif role == "tool":
            result = {"type": "tool_result", "tool_use_id": message["tool_call_id"], "content": content}
            add("user", [result])
    if "temperature" not in options:  # Thinking takes no temperature but 1.
        options = {"thinking": {"type": "enabled", "budget_tokens": 16000}, **options}
    body = {
        "model": model,
        "max_tokens": 32000,
        "stream": True,
        "messages": turns,
        "tools": [
            {"name": f["name"], "description": f["description"], "input_schema": f["parameters"]}
            for f in (t["function"] for t in tools)
        ],
        **options,
    }
    if system := system_text(messages):
        body["system"] = [{"type": "text", "text": system}]
    if marks:
        mark(body)
    return body


def mark(body: dict) -> None:
    """Cache the system prompt on its own and the prompt up to its end."""
    ephemeral = {"type": "ephemeral"}
    if body.get("system"):
        body["system"][-1]["cache_control"] = ephemeral
    if body["messages"]:
        body["messages"][-1]["content"][-1]["cache_control"] = ephemeral


def anthropic_parts(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    blocks = []
    for part in content:
        if part["type"] == "text":
            blocks.append({"type": "text", "text": part["text"]})
        else:
            kind, data = part["image_url"]["url"].removeprefix("data:").split(";base64,", 1)
            blocks.append({"type": "image", "source": {"type": "base64", "media_type": kind, "data": data}})
    return blocks


# Replies.


def read(api: str, events) -> Turn:
    """The reply in a stream of server-sent event data, up to its last event.
    Raise `StreamError` on an error event, and ConnectionError if the stream
    closed before any event."""
    turn, seen = Turn(), False
    for data in events:
        if data == "[DONE]":
            turn.done = True
        else:
            event = json.loads(data)
            seen = True
            if failure := error_of(api, event):
                raise StreamError(f"stream error: {failure}")
            {"chat": chat_event, "responses": responses_event, "anthropic": anthropic_event}[api](turn, event)
        # Chat's finish chunk comes before its usage chunk and `[DONE]`.
        if turn.done:
            break
    if not seen and not turn.done:
        raise ConnectionError("the stream closed before any event")
    return turn


def error_of(api: str, event: dict) -> str | None:
    if api == "chat":
        return event["error"].get("message") if "error" in event else None
    kind = event.get("type")
    if kind == "error":
        inner = event.get("error", event)
        return inner.get("message", "error")
    if kind == "response.failed":
        return ((event.get("response") or {}).get("error") or {}).get("message", "failed")
    return None


def chat_event(turn: Turn, event: dict) -> None:
    if not event.get("choices") and "--need-choices" in sys.argv:
        raise ValueError("a chunk with no choices")  # As a parser that indexes choices[0].
    for choice in event.get("choices", []):
        delta = choice.get("delta", {})
        turn.stop = choice.get("finish_reason") or turn.stop
        turn.text += delta.get("content") or ""
        turn.reasoning += delta.get("reasoning_content") or ""
        for part in delta.get("tool_calls") or []:
            call = turn.call(part["index"])
            # The ID and name come in the call's first chunk; some servers repeat them.
            call["id"] = call["id"] or part.get("id") or ""
            call["name"] = call["name"] or (part.get("function") or {}).get("name") or ""
            call["arguments"] += (part.get("function") or {}).get("arguments") or ""


def responses_event(turn: Turn, event: dict) -> None:
    kind = event.get("type", "")
    if kind == "response.output_text.delta":
        turn.text += event.get("delta", "")
    elif kind == "response.reasoning_summary_text.delta":
        turn.reasoning += event.get("delta", "")
    elif kind == "response.output_item.added" and event["item"].get("type") == "function_call":
        turn.call(event["output_index"], event["item"].get("call_id", ""), event["item"].get("name", ""))
    elif kind == "response.function_call_arguments.delta":
        turn.call(event["output_index"])["arguments"] += event.get("delta", "")
    elif kind == "response.output_item.done":
        item = event["item"]
        if item.get("type") == "reasoning":
            turn.items.append(item)
        elif item.get("type") == "function_call":
            turn.call(event["output_index"])["arguments"] = item.get("arguments", "")
    elif kind in ("response.completed", "response.incomplete"):
        turn.done, turn.stop = True, kind.removeprefix("response.")
        # The final response is whole: some servers send encrypted reasoning only here.
        if "--early-reasoning-only" not in sys.argv:
            final = {i.get("id"): i for i in event.get("response", {}).get("output") or []}
            for item in turn.items:
                item.update(final.get(item.get("id")) or {})


def anthropic_event(turn: Turn, event: dict) -> None:
    kind, position = event.get("type", ""), event.get("index")
    if kind == "content_block_start":
        anthropic_block(turn, position, dict(event["content_block"]))
    elif kind == "content_block_delta":
        delta = event.get("delta", {})
        thinking = next((b for b in turn.items if b.get("index") == position), None)
        if delta.get("type") == "text_delta":
            turn.text += delta.get("text", "")
        elif delta.get("type") == "input_json_delta":
            turn.call(position)["arguments"] += delta.get("partial_json", "")
        elif delta.get("type") == "thinking_delta" and thinking is not None:
            thinking["thinking"] += delta.get("thinking", "")
            turn.reasoning += delta.get("thinking", "")
        elif delta.get("type") == "signature_delta" and thinking is not None:
            thinking["signature"] += delta.get("signature", "")
    elif kind == "content_block_stop":
        for block in turn.items:
            if block.get("index") == position:
                del block["index"]
    elif kind == "message_delta":
        turn.stop = event.get("delta", {}).get("stop_reason") or turn.stop
    elif kind == "message_stop":
        turn.done = True


def anthropic_block(turn: Turn, position: int, block: dict) -> None:
    kind = block.get("type")
    if kind == "text":
        turn.text += block.get("text", "")
    elif kind == "thinking":
        turn.items.append({**block, "index": position})
        turn.reasoning += block.get("thinking", "")
    elif kind == "redacted_thinking":
        turn.items.append(block)
    elif kind == "tool_use":
        call = turn.call(position, block.get("id", ""), block.get("name", ""))
        # A stream starts the block with empty input and sends it in deltas.
        if block.get("input"):
            call["arguments"] = json.dumps(block["input"])
