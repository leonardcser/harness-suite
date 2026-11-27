"""OpenAI Chat Completions."""

import json
from itertools import zip_longest

from ..messages import Call, Error, Message, Reply, Request, Result, Text, Thinking, Tool
from . import openai_errors
from .common import (
    NAME_PATTERN,
    Frame,
    Rejection,
    append,
    data_url,
    fragments,
    frame,
    media,
    not_object,
    rejections,
    text_of,
    text_pieces,
    unsupported_image,
)

NAME = "chat"
PATH = "/v1/chat/completions"


def content_parts(content) -> list:
    if isinstance(content, str):
        return [Text(content)] if content else []
    parts = []
    for block in content or []:
        kind = block.get("type")
        if kind == "text":
            parts.append(Text(block.get("text", "")))
        elif kind == "image_url":
            url = block.get("image_url")
            found = data_url(url.get("url") if isinstance(url, dict) else url)
            parts.append(found or Text("[image reference]"))
        elif kind == "file":
            file = block.get("file", {})
            found = data_url(file.get("file_data", ""))
            parts.append(found or Text("[file reference]"))
        elif kind == "input_audio":
            parts.append(media("audio/" + block["input_audio"]["format"], ""))
    return parts


def parse(body: dict, headers: dict, store: dict) -> Request:
    messages: list[Message] = []
    system = []
    for item in body.get("messages", []):
        role = item.get("role")
        if role in {"system", "developer"}:
            system.append(text_of(item.get("content")))
        elif role == "user":
            append(messages, "user", content_parts(item.get("content")))
        elif role == "assistant":
            parts = []
            reasoning = item.get("reasoning_content") or item.get("reasoning")
            if not isinstance(reasoning, str):
                # OpenRouter's structured form: a list of reasoning parts.
                details = item.get("reasoning_details")
                parts_of = details if isinstance(details, list) else []
                reasoning = "".join(d.get("text") or d.get("summary") or "" for d in parts_of if isinstance(d, dict))
            if reasoning:
                parts.append(Thinking(reasoning))
            parts += content_parts(item.get("content"))
            for call in item.get("tool_calls") or []:
                function = call.get("function", {})
                parts.append(Call(call.get("id", ""), function.get("name", ""), function.get("arguments", "")))
            append(messages, "assistant", parts)
        elif role == "tool":
            parts = content_parts(item.get("content"))
            append(messages, "user", [Result(item.get("tool_call_id", ""), tuple(parts))])
    tools = [
        Tool(f.get("name", ""), f.get("description", ""), f.get("parameters", {}))
        for spec in body.get("tools") or []
        if (f := spec.get("function"))
    ]
    parameters = {
        "max_tokens": body.get("max_completion_tokens", body.get("max_tokens")),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "effort": body.get("reasoning_effort"),
        "service_tier": body.get("service_tier"),
        "tool_choice": body.get("tool_choice"),
        "parallel_tool_calls": body.get("parallel_tool_calls"),
    }
    return Request(
        NAME,
        body,
        headers,
        body.get("model"),
        bool(body.get("stream")),
        "\n".join(system),
        messages,
        tools,
        {k: v for k, v in parameters.items() if v is not None},
    )


MAX_TOOLS = 128  # OpenAI rejects a request with more tools.


def tool_count_problem(request: Request, store: dict) -> str | None:
    specs = request.raw.get("tools") or []
    if len(specs) > MAX_TOOLS:
        return (
            "Invalid 'tools': array too long. Expected an array with maximum length"
            f" {MAX_TOOLS}, but got an array with length {len(specs)} instead."
        )
    return None


def tool_name_problem(request: Request, store: dict) -> str | None:
    for index, spec in enumerate(request.raw.get("tools") or []):
        if not NAME_PATTERN.fullmatch((spec.get("function") or {}).get("name", "")):
            return (
                f"Invalid 'tools[{index}].function.name': string does not match pattern."
                " Expected a string that matches the pattern '^[a-zA-Z0-9_-]+$'."
            )
    return None


def tool_schema_problem(request: Request, store: dict) -> str | None:
    for spec in request.raw.get("tools") or []:
        function = spec.get("function") or {}
        if found := not_object(function.get("parameters"), function.get("name", "")):
            return found
    return None


def unanswered(waiting: list[str]) -> str:
    return (
        "An assistant message with 'tool_calls' must be followed by tool messages"
        " responding to each 'tool_call_id'. The following tool_call_ids did not"
        f" have response messages: {', '.join(waiting)}"
    )


def tool_pairs_problem(request: Request, store: dict) -> str | None:
    waiting: list[str] = []
    for item in request.raw.get("messages", []):
        role = item.get("role")
        if role == "tool":
            if item.get("tool_call_id") not in waiting:
                return (
                    "Invalid parameter: messages with role 'tool' must be a response"
                    " to a preceeding message with 'tool_calls'."
                )
            waiting.remove(item["tool_call_id"])
            continue
        if waiting:
            return unanswered(waiting)
        if role == "assistant":
            waiting = [call.get("id") for call in item.get("tool_calls") or []]
    return unanswered(waiting) if waiting else None


def reasoning_problem(request: Request, store: dict) -> str | None:
    """Providers that return `reasoning_content` (DeepSeek, Kimi, GLM,
    MiniMax) need it back, unchanged, while their tool calls run."""
    raw = request.raw.get("messages", [])
    last_turn = next((item for item in reversed(raw) if item.get("role") == "assistant"), {})
    open_turn = bool(raw) and raw[-1].get("role") == "tool"
    calls = [call.get("id") for call in last_turn.get("tool_calls") or []]
    expected = next((store["reasoning"][c] for c in calls if c in store.get("reasoning", {})), None)
    sent = last_turn.get("reasoning_content") or last_turn.get("reasoning") or ""
    if open_turn and expected and sent.strip() != expected.strip():
        return (
            "The `reasoning_content` of an assistant message with tool calls must be passed back unchanged"
            " in thinking mode."
        )
    return None


RULES = [
    ("image", lambda request, store: unsupported_image(request)),
    ("tool-count", tool_count_problem),
    ("tool-name", tool_name_problem),
    ("tool-schema", tool_schema_problem),
    ("tool-pairs", tool_pairs_problem),
    ("reasoning", reasoning_problem),
]


def problems(request: Request, store: dict) -> list[Rejection]:
    """Every rule of the real API that the request breaks."""
    return rejections(request, store, RULES)


def usage(request: Request, reply: Reply) -> dict:
    tokens = request.tokens
    return {
        "prompt_tokens": tokens,
        "completion_tokens": reply.output_tokens,
        "total_tokens": tokens + reply.output_tokens,
    }


def finish(reply: Reply) -> str:
    reasons = {"tool": "tool_calls", "end": "stop", "length": "length", "refusal": "content_filter", "error": "error"}
    return reasons[reply.stop_reason]


def remember(reply: Reply, store: dict) -> None:
    """Keep the reasoning of a tool turn: the client must send it back."""
    for call in reply.calls if reply.thinking else ():
        store.setdefault("reasoning", {})[call.id] = reply.thinking


def body(request: Request, reply: Reply, number: int, store: dict) -> dict:
    remember(reply, store)
    message: dict = {"role": "assistant", "content": reply.text or None}
    if reply.thinking:
        message["reasoning_content"] = reply.thinking
    if reply.calls:
        message["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments},
            }
            for call in reply.calls
        ]
    return {
        "id": f"chatcmpl-suite-{number}",
        "object": "chat.completion",
        "created": 1_760_000_000,
        "model": request.model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish(reply)}],
        "usage": usage(request, reply),
    }


def call_chunks(index: int, call: Call, mode: str) -> list[dict]:
    """The `tool_calls` entries of one call's chunks: its ID and name first,
    then its arguments in fragments. Some servers repeat the ID and name in
    every chunk, or send them blank after the first."""
    identity = {"id": call.id, "type": "function"}
    if mode == "whole":
        return [{"index": index, **identity, "function": {"name": call.name, "arguments": call.arguments}}]
    later = {
        "repeated-identity": (identity, {"name": call.name}),
        "blank-identity": ({"id": "", "type": "function"}, {"name": ""}),
    }.get(mode, ({}, {}))
    start = {"index": index, **identity, "function": {"name": call.name, "arguments": ""}}
    return [start] + [
        {"index": index, **later[0], "function": {**later[1], "arguments": part}}
        for part in fragments(call.arguments, (17, 9, 23))
    ]


def events(request: Request, reply: Reply, number: int, store: dict) -> list[Frame]:
    base = {
        "id": f"chatcmpl-suite-{number}",
        "object": "chat.completion.chunk",
        "created": 1_760_000_000,
        "model": request.model,
    }

    def chunk(delta: dict, reason=None, terminal=False) -> Frame:
        value = {**base, "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]}
        return frame(value, False, terminal)

    remember(reply, store)
    frames = [chunk({"role": "assistant", "content": ""})]
    frames += [chunk({"reasoning_content": part}) for part in fragments(reply.thinking)]
    frames += [chunk({"content": part}) for part in text_pieces(reply.text, reply.stream)]
    mode = reply.stream.calls
    pieces = [call_chunks(index, call, mode) for index, call in enumerate(reply.calls)]
    if mode == "interleaved":
        pieces = [[piece for piece in turn if piece] for turn in zip_longest(*pieces)]
    frames += [chunk({"tool_calls": [piece]}) for call in pieces for piece in call]
    frames.append(chunk({}, finish(reply), True))
    if (request.raw.get("stream_options") or {}).get("include_usage"):
        frames.append(frame({**base, "choices": [], "usage": usage(request, reply)}, False, True))
    frames.append(Frame("[DONE]", terminal=True))
    return frames


def error(err: Error, request: Request, window: int) -> tuple[int, dict, dict]:
    return openai_errors.response(err, request, window, "messages")


def stream_error(err: Error, request: Request, window: int) -> Frame:
    _, _, value = openai_errors.response(err, request, window, "messages")
    return Frame(json.dumps(value))


def azure_filter() -> list[Frame]:
    """The chunk Azure OpenAI sends first: no choices, only the prompt's
    content filter results."""
    return [
        frame(
            {
                "id": "",
                "object": "",
                "created": 0,
                "model": "",
                "choices": [],
                "prompt_filter_results": [{"prompt_index": 0, "content_filter_results": {}}],
            },
            False,
        )
    ]


def unknown(request: Request, number: int) -> list[Frame]:
    """A valid chunk a client may not expect: a delta with a field from a
    newer API version."""
    return [
        frame(
            {
                "id": f"chatcmpl-suite-{number}",
                "object": "chat.completion.chunk",
                "created": 1_760_000_000,
                "model": request.model,
                "choices": [{"index": 0, "delta": {"suite_future_field": {"x": 1}}, "finish_reason": None}],
            },
            False,
        ),
    ]
