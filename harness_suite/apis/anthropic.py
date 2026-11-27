"""Anthropic Messages."""

import itertools
import json
import re

from ..messages import (
    Call,
    Error,
    Message,
    Reply,
    Request,
    Result,
    Text,
    Thinking,
    Tool,
    opaque,
    signature,
)
from .common import IMAGE_FORMATS, Frame, Rejection, append, fragments, frame, media, rejections, text_of, text_pieces

NAME = "anthropic"
PATH = "/v1/messages"
TOOL_NAME = re.compile(r"[a-zA-Z0-9_-]{1,128}")
TOOL_ID = re.compile(r"[a-zA-Z0-9_-]+")
THINKING = {"thinking", "redacted_thinking"}
UNMODIFIED = "`thinking` or `redacted_thinking` blocks in the latest assistant message cannot be modified. These blocks must remain as they were in the original response."


def blocks(content) -> list[dict]:
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    return [block for block in content or [] if isinstance(block, dict)]


def part(block: dict):
    kind = block.get("type")
    if kind == "text":
        return Text(block.get("text", ""))
    if kind in {"image", "document"}:
        source = block.get("source", {})
        if source.get("type") == "base64":
            return media(source.get("media_type", ""), source.get("data", ""))
        if source.get("type") == "text":
            return Text(source.get("data", ""))
        return Text(f"[{kind} reference]")
    if kind == "thinking":
        return Thinking(block.get("thinking", ""), block.get("signature"))
    if kind == "redacted_thinking":
        return Thinking("", block.get("data"))
    if kind == "tool_use":
        return Call(block.get("id", ""), block.get("name", ""), json.dumps(block.get("input", {})))
    if kind == "tool_result":
        inner = block.get("content", "")
        parts = tuple(p for b in blocks(inner) if (p := part(b)) is not None)
        return Result(block.get("tool_use_id", ""), parts, bool(block.get("is_error")))
    return None


def parse(body: dict, headers: dict, store: dict) -> Request:
    messages: list[Message] = []
    for item in body.get("messages", []):
        parts = [p for block in blocks(item.get("content")) if (p := part(block)) is not None]
        append(messages, item.get("role", "user"), parts)
    tools = [
        Tool(t["name"], t.get("description", ""), t.get("input_schema", {}))
        for t in body.get("tools") or []
        if t.get("name") and "input_schema" in t
    ]
    asked = body.get("thinking") or {}
    parameters = {
        "max_tokens": body.get("max_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "effort": (body.get("output_config") or {}).get("effort"),
        "thinking": asked.get("type"),
        "thinking_budget": asked.get("budget_tokens"),
        "speed": body.get("speed"),
        "service_tier": body.get("service_tier"),
        "tool_choice": (body.get("tool_choice") or {}).get("type"),
        "beta": headers.get("anthropic-beta"),
    }
    return Request(
        NAME,
        body,
        headers,
        body.get("model"),
        bool(body.get("stream")),
        text_of(body.get("system") or ""),
        messages,
        tools,
        {k: v for k, v in parameters.items() if v is not None},
        reasons=thinking(body),
    )


def cache_marks(raw: dict) -> list[tuple[str, int]]:
    """How many `cache_control` marks each part of a request has."""

    def count(content) -> int:
        return sum(1 for b in blocks(content) if b.get("cache_control"))

    messages = raw.get("messages", [])
    return [
        ("the request", 1 if raw.get("cache_control") else 0),
        ("tools", sum(1 for t in raw.get("tools") or [] if t.get("cache_control"))),
        ("system", count(raw.get("system"))),
        ("earlier messages", sum(count(m.get("content")) for m in messages[:-1])),
        ("the last message", count(messages[-1].get("content")) if messages else 0),
    ]


def ttls(raw: dict) -> list[str]:
    """The time to live of each `cache_control` mark, in the order the API
    reads the prompt: tools, system, messages."""
    found = [t.get("cache_control") for t in raw.get("tools") or []]
    found += [b.get("cache_control") for b in blocks(raw.get("system"))]
    found += [b.get("cache_control") for m in raw.get("messages", []) for b in blocks(m.get("content"))]
    return [mark.get("ttl", "5m") for mark in found if isinstance(mark, dict)]


def turns(messages: list) -> list[dict]:
    """The messages as the API reads them: it joins consecutive messages of the
    same role into one turn."""
    joined: list[dict] = []
    for item in messages:
        if joined and joined[-1]["role"] == item.get("role"):
            joined[-1]["content"] += blocks(item.get("content"))
        else:
            joined.append({"role": item.get("role"), "content": blocks(item.get("content"))})
    return joined


def thinking(body: dict) -> bool:
    return (body.get("thinking") or {}).get("type") in {"enabled", "adaptive"}


INTERLEAVED = "interleaved-thinking-2025-05-14"


def interleaved(request: Request) -> bool:
    """Whether the request asks for thinking between tool calls."""
    betas = next((v for k, v in request.headers.items() if k.lower() == "anthropic-beta"), "")
    return INTERLEAVED in [beta.strip() for beta in betas.split(",")]


IMAGE_BYTES = 10 * 1024**2  # Of base64, on the Claude API; Bedrock takes 5 MB.
IMAGE_SIDE = 8000
# A request with more than 20 images has a stricter limit on every image's
# sides. The docs say 3000 px stays under it on every platform.
MANY_IMAGES, MANY_SIDE = 20, 3000


def image_problem(request: Request) -> str | None:
    """Anthropic's error for an image it does not take: another format, bytes
    that do not match the declared type, or one over its size limits."""
    for image in request.images:
        if image.media_type not in {f"image/{name}" for name in IMAGE_FORMATS}:
            return (
                "messages: image.source.base64.media_type: Input should be"
                " 'image/jpeg', 'image/png', 'image/gif' or 'image/webp'"
            )
        if image.format != image.media_type.removeprefix("image/"):
            return f"messages: Image does not match the provided media type {image.media_type}"
        encoded = 4 * -(-image.size // 3)
        if encoded > IMAGE_BYTES:
            return f"messages: image exceeds 10 MB maximum: {encoded} bytes > {IMAGE_BYTES} bytes"
        if max(image.width, image.height) > IMAGE_SIDE:
            return f"messages: image dimensions exceed max allowed size: {IMAGE_SIDE} pixels"
    if len(request.images) > MANY_IMAGES and any(max(i.width, i.height) > MANY_SIDE for i in request.images):
        return (
            "messages: At least one of the image dimensions exceed max allowed size for"
            f" many-image requests: {MANY_SIDE} pixels"
        )
    return None


def text_problem(block: dict) -> str | None:
    """What the API says of a text block in a message: it takes neither an
    empty one nor one of whitespace alone."""
    if block.get("type") != "text":
        return None
    if not block.get("text"):
        return "messages: text content blocks must be non-empty"
    if not str(block["text"]).strip():
        return "messages: text content blocks must contain non-whitespace text"
    return None


def sampling_problem(request: Request, store: dict) -> str | None:
    if not thinking(request.raw):
        return None
    if request.raw.get("temperature") not in (None, 1):
        return "`temperature` may only be set to 1 when thinking is enabled."
    if request.raw.get("top_p") is not None and request.raw["top_p"] < 0.95:
        return "`top_p` must be greater than or equal to 0.95 or unset when thinking is enabled."
    budget = (request.raw.get("thinking") or {}).get("budget_tokens")
    if isinstance(budget, int) and budget < 1024:
        return "thinking.enabled.budget_tokens: Input should be greater than or equal to 1024"
    # Interleaved thinking lets the budget cover a whole turn of tool calls.
    if isinstance(budget, int) and budget >= (request.raw.get("max_tokens") or 0) and not interleaved(request):
        return "`max_tokens` must be greater than `thinking.budget_tokens`."
    return None


def cache_problem(request: Request, store: dict) -> str | None:
    marks = sum(n for _, n in cache_marks(request.raw))
    if marks > 4:
        return f"A maximum of 4 blocks with cache_control may be provided. Found {marks}."
    lives = ttls(request.raw)
    if "1h" in lives and "5m" in lives[: len(lives) - lives[::-1].index("1h")]:
        return "A 'ttl'='1h' cache_control block must not come after a 'ttl'='5m' cache_control block."
    return None


def message_blocks(request: Request):
    """Each block of the messages, as (message index, block index, block)."""
    for index, item in enumerate(turns(request.raw.get("messages", []))):
        for position, block in enumerate(blocks(item.get("content"))):
            yield index, position, block


def tool_name_problem(request: Request, store: dict) -> str | None:
    names: set[str] = set()
    for index, tool in enumerate(request.raw.get("tools") or []):
        if "input_schema" in tool and not TOOL_NAME.fullmatch(tool.get("name", "")):
            return f"tools.{index}.custom.name: String should match pattern '^[a-zA-Z0-9_-]{{1,128}}$'"
        if tool.get("name") in names:
            return "tools: Tool names must be unique."
        names.add(tool.get("name"))
    for index, position, block in message_blocks(request):
        if block.get("type") == "tool_use" and len(block.get("name", "")) > 200:
            return f"messages.{index}.content.{position}.tool_use.name: String should have at most 200 characters"
    return None


def tool_schema_problem(request: Request, store: dict) -> str | None:
    for index, tool in enumerate(request.raw.get("tools") or []):
        if "input_schema" in tool and (tool["input_schema"] or {}).get("type") != "object":
            return f"tools.{index}.custom.input_schema.type: Input should be 'object'"
    return None


def text_rule(request: Request, store: dict) -> str | None:
    for block in blocks(request.raw.get("system")):
        if block.get("type") == "text" and not block.get("text"):
            return "system: text content blocks must be non-empty"
    for _, _, block in message_blocks(request):
        if found := text_problem(block):
            return found
        if block.get("type") == "tool_result" and isinstance(block.get("content"), list):
            for inner in blocks(block["content"]):
                if inner.get("type") == "text" and not inner.get("text"):
                    return "messages: text content blocks must be non-empty"
    return None


def tool_id_problem(request: Request, store: dict) -> str | None:
    seen: set[str] = set()
    for index, position, block in message_blocks(request):
        if block.get("type") != "tool_use":
            continue
        where = f"messages.{index}.content.{position}"
        if not TOOL_ID.fullmatch(block.get("id", "")):
            return f"{where}.tool_use.id: String should match pattern '^[a-zA-Z0-9_-]+$'"
        if block["id"] in seen:
            return f"{where}: `tool_use` ids must be unique"
        seen.add(block["id"])
    return None


def tool_pairs_problem(request: Request, store: dict) -> str | None:
    raw = turns(request.raw.get("messages", []))
    for index, item in enumerate(raw):
        here = blocks(item.get("content"))
        if item.get("role") == "user":
            results = [b for b in here if b.get("type") == "tool_result"]
            if results and here[: len(results)] != results:
                return (
                    f"messages.{index}: `tool_result` blocks must come first in the"
                    " user message content, before any other content"
                )
            previous = blocks(raw[index - 1].get("content")) if index else []
            expected = {b.get("id") for b in previous if b.get("type") == "tool_use"}
            for position, block in enumerate(results):
                if block.get("tool_use_id") not in expected:
                    return (
                        f"messages.{index}.content.{position}: unexpected `tool_use_id`"
                        f" found in `tool_result` blocks: {block.get('tool_use_id')}."
                        " Each `tool_result` block must have a corresponding `tool_use`"
                        " block in the previous message."
                    )
        uses = [b for b in here if b.get("type") == "tool_use"]
        if uses and index + 1 < len(raw):
            following = blocks(raw[index + 1].get("content"))
            answered = {b.get("tool_use_id") for b in following if b.get("type") == "tool_result"}
            if missing := [b["id"] for b in uses if b["id"] not in answered]:
                return (
                    f"messages.{index}: `tool_use` ids were found without `tool_result`"
                    f" blocks immediately after: {', '.join(missing)}. Each `tool_use`"
                    " block must have a corresponding `tool_result` block in the next"
                    " message."
                )
    return None


def reasoning_problem(request: Request, store: dict) -> str | None:
    for index, position, block in message_blocks(request):
        where = f"messages.{index}.content.{position}"
        if block.get("type") == "thinking":
            if "signature" not in block:
                return f"{where}.thinking.signature: Field required"
            # An omitted block's text is ignored; only its signature counts.
            omitted = block["signature"] in store.get("opaque", set())
            if not omitted and block["signature"] != signature(block.get("thinking", "")):
                return f"{where}: Invalid `signature` in `thinking` block"
        if block.get("type") == "redacted_thinking" and block.get("data") not in store.get("opaque", set()):
            return f"{where}: {UNMODIFIED}"
    raw = turns(request.raw.get("messages", []))
    thinking_on = thinking(request.raw)
    # Adaptive thinking drops the rule that a final turn starts with thinking.
    manual = (request.raw.get("thinking") or {}).get("type") == "enabled"
    for index, item in enumerate(raw):
        here = blocks(item.get("content"))
        uses = [b for b in here if b.get("type") == "tool_use"]
        if not uses or index + 2 != len(raw):
            continue
        issued = next((store["turns"][b["id"]] for b in uses if b["id"] in store.get("turns", {})), None)
        leading = [key(b) for b in itertools.takewhile(lambda b: b.get("type") in THINKING, here)]
        if thinking_on and issued and leading and leading != issued:
            return f"messages.{index}: {UNMODIFIED}"
        if manual and issued and here[0].get("type") not in THINKING:
            return (
                f"messages.{index}.content.0.type: Expected `thinking` or"
                f" `redacted_thinking`, but found `{here[0].get('type')}`. When"
                " `thinking` is enabled, a final `assistant` message must start"
                " with a thinking block (preceeding the lastmost set of `tool_use`"
                " and `tool_result` blocks)."
            )
    return None


RULES = [
    ("image", lambda request, store: image_problem(request)),
    ("sampling", sampling_problem),
    ("cache", cache_problem),
    ("tool-name", tool_name_problem),
    ("tool-schema", tool_schema_problem),
    ("text", text_rule),
    ("tool-id", tool_id_problem),
    ("reasoning", reasoning_problem),
    ("tool-pairs", tool_pairs_problem),
]


def problems(request: Request, store: dict) -> list[Rejection]:
    """Every rule of the real API that the request breaks."""
    return rejections(request, store, RULES)


def thinking_block(reply: Reply) -> dict:
    if reply.thinking_form == "redacted":
        return {"type": "redacted_thinking", "data": opaque(reply.thinking)}
    if reply.thinking_form == "omitted":
        return {"type": "thinking", "thinking": "", "signature": opaque(reply.thinking)}
    return {"type": "thinking", "thinking": reply.thinking, "signature": signature(reply.thinking)}


def key(block: dict) -> tuple[str, str]:
    """What identifies a thinking block: its signature, or a redacted block's data."""
    if block.get("type") == "redacted_thinking":
        return ("redacted_thinking", block.get("data", ""))
    return ("thinking", block.get("signature", ""))


def content(reply: Reply) -> list[dict]:
    values = []
    if reply.thinking:
        values.append(thinking_block(reply))
    if reply.text:
        values.append({"type": "text", "text": reply.text})
    for call in reply.calls:
        values.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.args() or {}})
    return values


def stop(reply: Reply) -> str:
    reasons = {
        "tool": "tool_use",
        "end": "end_turn",
        "length": "max_tokens",
        "refusal": "refusal",
        "window": "model_context_window_exceeded",
    }
    return reasons[reply.stop_reason]


def remember(reply: Reply, store: dict) -> None:
    """Keep the thinking blocks of a reply: the API checks them when they return."""
    if not reply.thinking:
        return
    block = thinking_block(reply)
    store.setdefault("opaque", set()).add(block.get("data") or block["signature"])
    for call in reply.calls:
        store.setdefault("turns", {})[call.id] = [key(block)]


def body(request: Request, reply: Reply, number: int, store: dict) -> dict:
    remember(reply, store)
    return {
        "id": f"msg_suite_{number}",
        "type": "message",
        "role": "assistant",
        "model": request.model,
        "content": content(reply),
        "stop_reason": stop(reply),
        "stop_sequence": None,
        "usage": usage(request, reply, reply.output_tokens),
    }


def usage(request: Request, reply: Reply, output: int) -> dict:
    return {
        "input_tokens": request.tokens,
        "cache_creation_input_tokens": 0,
        "cache_read_input_tokens": 0,
        "output_tokens": output,
    }


def events(request: Request, reply: Reply, number: int, store: dict) -> list[Frame]:
    remember(reply, store)
    start = {**body(request, reply, number, store), "content": [], "stop_reason": None}
    start["usage"] = usage(request, reply, 1)
    values = [{"type": "message_start", "message": start}, {"type": "ping"}]
    for index, block in enumerate(content(reply)):
        kind = block["type"]
        if kind == "redacted_thinking":
            opening, deltas = block, []
        elif kind == "thinking":
            opening = {"type": "thinking", "thinking": "", "signature": ""}
            deltas = [{"type": "thinking_delta", "thinking": p} for p in fragments(block["thinking"])]
            deltas.append({"type": "signature_delta", "signature": block["signature"]})
        elif kind == "text":
            opening = {"type": "text", "text": ""}
            deltas = [{"type": "text_delta", "text": p} for p in text_pieces(block["text"], reply.stream)]
        else:
            opening = {**block, "input": {}}
            call = reply.calls[index - len(content(reply)) + len(reply.calls)]
            deltas = [{"type": "input_json_delta", "partial_json": p} for p in fragments(call.arguments, (17, 9, 23))]
        values.append({"type": "content_block_start", "index": index, "content_block": opening})
        values += [{"type": "content_block_delta", "index": index, "delta": d} for d in deltas]
        values.append({"type": "content_block_stop", "index": index})
    frames = [frame(value, True) for value in values]
    frames.append(
        frame(
            {
                "type": "message_delta",
                "delta": {"stop_reason": stop(reply), "stop_sequence": None},
                "usage": usage(request, reply, reply.output_tokens),
            },
            True,
            terminal=True,
        )
    )
    frames.append(frame({"type": "message_stop"}, True, terminal=True))
    return frames


def headers(err: Error, status: int) -> dict:
    # The API tells its clients whether to retry: not after a client error,
    # such as an invalid key, unless it is a rate limit.
    retry = status >= 500 or status in (408, 409, 429)
    values = {"request-id": "req_suite", "x-should-retry": "true" if retry else "false"}
    if err.retry_after is not None:
        values["retry-after"] = err.retry_after
    if err.retry_after_ms is not None:
        values["retry-after-ms"] = err.retry_after_ms
    if err.kind == "rate":
        values["anthropic-ratelimit-requests-limit"] = "50"
        values["anthropic-ratelimit-requests-remaining"] = "0"
    return values


def envelope(err: Error, request: Request, window: int) -> tuple[int, dict]:
    status, kind, message = {
        "rate": (
            429,
            "rate_limit_error",
            "This request would exceed the rate limit for your organization of 50"
            " requests per minute. For details, refer to:"
            " https://docs.claude.com/en/api/rate-limits. You can see the response"
            " headers for current usage. Please reduce the prompt length or the"
            " maximum tokens requested, or try again later.",
        ),
        "server": (500, "api_error", "Internal server error"),
        "overloaded": (529, "overloaded_error", "Overloaded"),
        "auth": (401, "authentication_error", "invalid x-api-key"),
        "quota": (
            400,
            "invalid_request_error",
            "Your credit balance is too low to access the Anthropic API. Please go to"
            " Plans & Billing to upgrade or purchase credits.",
        ),
        "context": (
            400,
            "invalid_request_error",
            f"prompt is too long: {request.tokens} tokens > {window} maximum",
        ),
        "invalid": (400, "invalid_request_error", "Invalid request."),
        "model": (404, "not_found_error", f"model: {request.model}"),
        "too-large": (413, "request_too_large", "Request exceeds the maximum allowed number of bytes."),
    }[err.kind]
    value = {
        "type": "error",
        "error": {"type": kind, "message": err.message or message},
        "request_id": "req_suite",
    }
    return status, value


def error(err: Error, request: Request, window: int) -> tuple[int, dict, dict]:
    status, value = envelope(err, request, window)
    return status, headers(err, status), value


def stream_error(err: Error, request: Request, window: int) -> Frame:
    _, value = envelope(err, request, window)
    return frame(value, True)


def unknown(request: Request, number: int) -> list[Frame]:
    """An event type from a newer API version, which clients must ignore."""
    return [frame({"type": "suite_future_event", "data": {"x": 1}}, True)]
