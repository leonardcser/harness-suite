"""OpenAI Responses, with stored responses for `previous_response_id` and
item references, unless a request says `store: false`."""

import json
from itertools import zip_longest

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
from . import openai_errors
from .common import (
    NAME_PATTERN,
    Frame,
    Rejection,
    append,
    data_url,
    fragments,
    frame,
    not_object,
    rejections,
    text_of,
    text_pieces,
    unsupported_image,
)

NAME = "responses"
PATH = "/v1/responses"


# Stops that end a response as incomplete, and the reason the API gives.
INCOMPLETE = {"length": "max_output_tokens", "refusal": "content_filter"}


def content_parts(content) -> list:
    if isinstance(content, str):
        return [Text(content)] if content else []
    parts = []
    for block in content or []:
        kind = block.get("type")
        if kind in {"input_text", "output_text", "text", "refusal"}:
            parts.append(Text(block.get("text", block.get("refusal", ""))))
        elif kind == "input_image":
            parts.append(data_url(block.get("image_url", "")) or Text("[image reference]"))
        elif kind == "input_file":
            parts.append(data_url(block.get("file_data", "")) or Text("[file reference]"))
    return parts


def tool_list(specs: list, namespace: str | None = None) -> list[Tool]:
    tools = []
    for spec in specs:
        if spec.get("type") == "namespace":
            inner = spec["name"] if namespace is None else f"{namespace}.{spec['name']}"
            tools += tool_list(spec.get("tools", []), inner)
        elif spec.get("type", "function") == "function" and spec.get("name"):
            name = spec["name"] if namespace is None else f"{namespace}.{spec['name']}"
            tools.append(Tool(name, spec.get("description", ""), spec.get("parameters", {})))
    return tools


def history(body: dict, store: dict) -> list[dict]:
    incoming = body.get("input", [])
    if isinstance(incoming, str):
        incoming = [{"role": "user", "content": incoming}]
    previous = body.get("previous_response_id")
    items = list(store.get("responses", {}).get(previous, [])) if previous else []
    for item in incoming:
        if item.get("type") == "item_reference":
            item = store.get("items", {}).get(item.get("id"), item)
        items.append(item)
    return items


def parse(body: dict, headers: dict, store: dict) -> Request:
    messages: list[Message] = []
    system = [text_of(body.get("instructions") or "")]
    for item in history(body, store):
        kind = item.get("type", "message")
        if kind == "message":
            role = item.get("role")
            if role in {"system", "developer"}:
                system.append(text_of(item.get("content")))
            elif role in {"user", "assistant"}:
                append(messages, role, content_parts(item.get("content")))
        elif kind == "function_call":
            name = item.get("name", "")
            if item.get("namespace"):
                name = f"{item['namespace']}.{name}"
            append(messages, "assistant", [Call(item.get("call_id", ""), name, item.get("arguments", ""))])
        elif kind == "function_call_output":
            output = item.get("output", "")
            parts = content_parts(output) if isinstance(output, list) else [Text(output)]
            append(messages, "user", [Result(item.get("call_id", ""), tuple(parts))])
        elif kind == "reasoning":
            summary = "".join(part.get("text", "") for part in item.get("summary") or [])
            # Reasoning the server stored is known by its ID, with or without its content.
            sealed = item.get("encrypted_content") or store.get("sealed", {}).get(item.get("id"))
            append(messages, "assistant", [Thinking(summary, sealed)])
    reasoning = body.get("reasoning") or {}
    parameters = {
        "max_tokens": body.get("max_output_tokens"),
        "temperature": body.get("temperature"),
        "top_p": body.get("top_p"),
        "effort": reasoning.get("effort"),
        "service_tier": body.get("service_tier"),
        "tool_choice": body.get("tool_choice"),
        "parallel_tool_calls": body.get("parallel_tool_calls"),
        "store": body.get("store"),
    }
    return Request(
        NAME,
        body,
        headers,
        body.get("model"),
        bool(body.get("stream")),
        "\n".join(part for part in system if part),
        messages,
        tool_list(body.get("tools") or []),
        {k: v for k, v in parameters.items() if v is not None},
        body.get("previous_response_id"),
        history(body, store),
    )


def tool_name_problem(request: Request, store: dict) -> str | None:
    for tool in request.tools:
        if not NAME_PATTERN.fullmatch(tool.name.rsplit(".", 1)[-1]):
            return "Invalid 'tools.name': string does not match pattern '^[a-zA-Z0-9_-]+$'."
    return None


def tool_schema_problem(request: Request, store: dict) -> str | None:
    for spec in request.raw.get("tools") or []:
        if spec.get("type") == "function" and (found := not_object(spec.get("parameters"), spec.get("name", ""))):
            return found
    return None


def incoming(request: Request) -> list[dict]:
    items = request.raw.get("input", [])
    return items if isinstance(items, list) else []


def by_id(item: dict) -> bool:
    """A reference, or reasoning that only its ID can bring back. Reasoning
    with neither an ID nor encrypted content is left to the API."""
    return item.get("type") == "item_reference" or (
        item.get("type") == "reasoning" and not item.get("encrypted_content") and bool(item.get("id"))
    )


def missing_item(item: dict) -> str:
    return (
        f"Item with id '{item.get('id')}' not found. Items are not persisted when `store` is set to"
        " false. Try again with `store` set to true, or remove this item from your input."
    )


def store_problem(request: Request, store: dict) -> str | None:
    previous = request.raw.get("previous_response_id")
    if previous and previous not in store.get("responses", {}):
        return f"Previous response with id '{previous}' not found."
    for item in incoming(request):
        if item.get("type") == "item_reference" and item.get("id") not in store.get("items", {}):
            return missing_item(item)
    return None


def tool_pairs_problem(request: Request, store: dict) -> str | None:
    items = history(request.raw, store)
    calls = [i.get("call_id") for i in items if i.get("type") == "function_call"]
    outputs = [i.get("call_id") for i in items if i.get("type") == "function_call_output"]
    for call_id in outputs:
        if call_id not in calls:
            return f"No tool call found for function call output with call_id {call_id}."
    for call_id in calls:
        if call_id not in outputs:
            return f"No tool output found for function call {call_id}."
    return None


def reasoning_problem(request: Request, store: dict) -> str | None:
    for item in incoming(request):
        if item.get("type") == "reasoning" and by_id(item) and item.get("id") not in store.get("items", {}):
            return missing_item(item)
    items = history(request.raw, store)
    for item in items:
        sealed = item.get("encrypted_content")
        if item.get("type") == "reasoning" and sealed and sealed not in store.get("encrypted", set()):
            return f"The encrypted content for item {item.get('id')} could not be verified."
    present = {item.get("id") for item in items}
    for item in incoming(request):
        needed = store.get("requires", {}).get(item.get("id"))
        if item.get("type") == "function_call" and needed and needed not in present:
            return (
                f"Item '{item['id']}' of type 'function_call' was provided without its"
                f" required 'reasoning' item: '{needed}'."
            )
    return None


RULES = [
    ("image", lambda request, store: unsupported_image(request)),
    ("tool-name", tool_name_problem),
    ("tool-schema", tool_schema_problem),
    ("store", store_problem),
    ("reasoning", reasoning_problem),
    ("tool-pairs", tool_pairs_problem),
]


def problems(request: Request, store: dict) -> list[Rejection]:
    """Every rule of the real API that the request breaks."""
    return rejections(request, store, RULES)


def usage(request: Request, reply: Reply) -> dict:
    tokens = request.tokens
    return {
        "input_tokens": tokens,
        "input_tokens_details": {"cached_tokens": 0},
        "output_tokens": reply.output_tokens,
        "output_tokens_details": {"reasoning_tokens": 8 if reply.thinking else 0},
        "total_tokens": tokens + reply.output_tokens,
    }


def namespace_of(request: Request, name: str) -> str | None:
    tool = next((t for t in request.tools if t.name == name and "." in t.name), None)
    return tool.name.rsplit(".", 1)[0] if tool else None


def output(request: Request, reply: Reply, number: int, store: dict) -> list[dict]:
    items = []
    incomplete = reply.stop_reason in INCOMPLETE
    reasoning_id = None
    if reply.thinking:
        reasoning_id = f"rs_suite_{number}"
        shown = reply.thinking_form == "shown"
        # Hidden reasoning has no summary, and the summary comes only when the
        # request asks for one (`reasoning.summary`). The encrypted content
        # comes only on request, and without stored responses it is the only
        # way back. The server keeps it for each item, as its own copy.
        encrypted = signature(reply.thinking) if shown else opaque(reply.thinking)
        store.setdefault("encrypted", set()).add(encrypted)
        summarized = shown and bool((request.raw.get("reasoning") or {}).get("summary"))
        item = {
            "id": reasoning_id,
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": reply.thinking}] if summarized else [],
        }
        if "reasoning.encrypted_content" in (request.raw.get("include") or []):
            item["encrypted_content"] = encrypted
        store.setdefault("sealed", {})[reasoning_id] = encrypted
        items.append(item)
    if reply.text:
        items.append(
            {
                "id": f"msg_suite_{number}",
                "type": "message",
                "role": "assistant",
                "status": "incomplete" if incomplete and not reply.calls else "completed",
                "content": [{"type": "output_text", "text": reply.text, "annotations": []}],
            }
        )
    for index, call in enumerate(reply.calls):
        namespace = namespace_of(request, call.name)
        item = {
            "id": f"fc_suite_{number}_{index}",
            "type": "function_call",
            "status": "incomplete" if incomplete else "completed",
            "call_id": call.id,
            "name": call.name[len(namespace) + 1 :] if namespace else call.name,
            "arguments": call.arguments,
        }
        if namespace:
            item["namespace"] = namespace
        if reasoning_id:
            store.setdefault("requires", {})[item["id"]] = reasoning_id
        items.append(item)
    if request.raw.get("store") is not False:
        for item in items:
            store.setdefault("items", {})[item["id"]] = item
    return items


def response(request: Request, reply: Reply, number: int, items: list, status: str) -> dict:
    return {
        "id": f"resp_suite_{number}",
        "object": "response",
        "created_at": 1_760_000_000,
        "status": status,
        "model": request.model,
        "output": items,
        "error": None,
        "incomplete_details": {"reason": INCOMPLETE[reply.stop_reason]} if status == "incomplete" else None,
        "usage": usage(request, reply) if status != "in_progress" else None,
    }


def remember(request: Request, number: int, items: list, store: dict) -> None:
    """Keep a response for `previous_response_id`, unless the request says
    `store: false`."""
    if request.raw.get("store") is False:
        return
    store.setdefault("responses", {})[f"resp_suite_{number}"] = [
        *history(request.raw, store),
        *items,
    ]


def body(request: Request, reply: Reply, number: int, store: dict) -> dict:
    items = output(request, reply, number, store)
    remember(request, number, items, store)
    status = "incomplete" if reply.stop_reason in INCOMPLETE else "completed"
    return response(request, reply, number, items, status)


def events(request: Request, reply: Reply, number: int, store: dict) -> list[Frame]:
    items = output(request, reply, number, store)
    remember(request, number, items, store)
    events = [
        {"type": "response.created", "response": response(request, reply, number, [], "in_progress")},
        {"type": "response.in_progress", "response": response(request, reply, number, [], "in_progress")},
    ]
    mode = reply.stream.calls
    calls = []
    for index, item in enumerate(items):
        where = {"output_index": index, "item_id": item["id"]}
        if item["type"] == "reasoning":
            done = {k: v for k, v in item.items() if k != "encrypted_content"} if reply.stream.late_reasoning else item
            events.append(
                {"type": "response.output_item.added", "output_index": index, "item": {**done, "summary": []}}
            )
            for summary in item["summary"]:
                part = {**where, "summary_index": 0}
                events.append(
                    {
                        "type": "response.reasoning_summary_part.added",
                        **part,
                        "part": {"type": "summary_text", "text": ""},
                    }
                )
                events += [
                    {"type": "response.reasoning_summary_text.delta", **part, "delta": piece}
                    for piece in fragments(summary["text"])
                ]
                events.append({"type": "response.reasoning_summary_text.done", **part, "text": summary["text"]})
                events.append({"type": "response.reasoning_summary_part.done", **part, "part": summary})
            events.append({"type": "response.output_item.done", "output_index": index, "item": done})
            continue
        elif item["type"] == "message":
            events.append(
                {
                    "type": "response.output_item.added",
                    "output_index": index,
                    "item": {**item, "status": "in_progress", "content": []},
                }
            )
            part = {**where, "content_index": 0}
            events.append(
                {
                    "type": "response.content_part.added",
                    **part,
                    "part": {"type": "output_text", "text": "", "annotations": []},
                }
            )
            events += [
                {"type": "response.output_text.delta", **part, "delta": piece}
                for piece in text_pieces(reply.text, reply.stream)
            ]
            events.append({"type": "response.output_text.done", **part, "text": reply.text})
            events.append({"type": "response.content_part.done", **part, "part": item["content"][0]})
        else:
            # A call's events stay apart, to interleave with its siblings'.
            pieces = [] if mode == "whole" else fragments(item["arguments"], (17, 9, 23))
            calls.append(
                [
                    {
                        "type": "response.output_item.added",
                        "output_index": index,
                        "item": {**item, "status": "in_progress", "arguments": ""},
                    },
                    *({"type": "response.function_call_arguments.delta", **where, "delta": p} for p in pieces),
                    {"type": "response.function_call_arguments.done", **where, "arguments": item["arguments"]},
                    {"type": "response.output_item.done", "output_index": index, "item": item},
                ]
            )
            continue
        events.append({"type": "response.output_item.done", "output_index": index, "item": item})
    if mode == "interleaved":
        calls = [[event for event in turn if event] for turn in zip_longest(*calls)]
    events += [event for call in calls for event in call]
    status = "incomplete" if reply.stop_reason in INCOMPLETE else "completed"
    final = {"type": f"response.{status}", "response": response(request, reply, number, items, status)}
    frames = [frame(event, True) for event in events]
    frames.append(frame(final, True, terminal=True))
    for sequence, value in enumerate(frames):
        frames[sequence] = sequenced(value, sequence)
    return frames


def sequenced(value: Frame, number: int) -> Frame:
    data = json.loads(value.data)
    data["sequence_number"] = number
    return Frame(json.dumps(data, ensure_ascii=False), value.event, value.terminal)


def error(err: Error, request: Request, window: int) -> tuple[int, dict, dict]:
    return openai_errors.response(err, request, window, "input")


def stream_error(err: Error, request: Request, window: int) -> Frame:
    _, _, value = openai_errors.response(err, request, window, "input")
    code = value["error"]["code"] or "server_error"
    failed = {
        "type": "response.failed",
        "response": {
            "id": "resp_suite_failed",
            "object": "response",
            "status": "failed",
            "output": [],
            "error": {"code": code, "message": value["error"]["message"]},
            "incomplete_details": None,
            "usage": None,
        },
    }
    return frame(failed, True, terminal=True)


def unknown(request: Request, number: int) -> list[Frame]:
    """An event type from a newer API version, which clients must ignore."""
    return [frame({"type": "response.suite_future_event", "data": {"x": 1}}, True)]
