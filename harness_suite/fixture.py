"""The suite's tool: an MCP server, or a shell command for native tools.

    python3 fixture.py mcp DIR [NAME]          an MCP server on stdio
    python3 fixture.py http DIR NAME           an MCP server on Streamable HTTP
    python3 fixture.py sse DIR NAME            an MCP server on the older HTTP with SSE
    python3 fixture.py run DIR ARGUMENTS_JSON  one call, as a shell command

Standard library only, so it runs unchanged inside any agent's container.
Every call is logged to DIR/calls.jsonl with its exact arguments, so checks
can count side effects and compare arguments without trusting the agent. A
slow call also logs a beat every half second while it runs, and an end.

The MCP server logs every message it gets as an `rpc` event, with its name.
A scenario can change how each server behaves with a profile in
DIR/servers.json, by server name (see `PROFILE`). An HTTP server listens on a
free port of 127.0.0.1 and writes it to DIR/NAME.port.
"""

import base64
import json
import os
import queue
import random
import re
import select
import signal
import struct
import sys
import threading
import time
import uuid
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

DESCRIPTION = (
    "Harness Suite fixture. Returns deterministic text, errors, side effects,"
    " images and PDFs for the scripted test model."
)
# The rule that the suite puts in the workspace's instruction files.
RULE_BEFORE = "SUITE_RULE_FIRST"
API_KEY = "suite-placeholder"  # The key every recipe gives the agent.
EXIT_STATUS = 173  # The `mixed` command's status: no other part of its output holds the number.
# Kinds for both tools, then kinds only a shell command or only MCP can give.
KINDS = ("text", "error", "effect", "echo", "slow", "image", "two-images", "killed", "empty", "escapes")
SHELL_KINDS = ("stdin", "binary", "background", "mixed", "inspect", "prompt", "pager", "flood", "long")
# Content only MCP results carry (`CONTENT`), and a broken answer.
MCP_KINDS = (
    "protocol-error",
    "resource-link",
    "embedded-text",
    "audio",
    "pdf",
    "blob",
    "json",
    "image-resource",
    "user-only",
    "lone-surrogate",
)
SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": [*KINDS, *SHELL_KINDS, *MCP_KINDS]},
        "token": {"type": "string"},
        "size": {"type": "integer", "minimum": 0},
        "style": {"type": "string", "enum": ["dense", "lines", "unicode", "failing"]},
        "value": {"type": "string"},
        "seconds": {"type": "number", "minimum": 0},
    },
    "required": ["kind", "token"],
}
# A schema with references, unions and nesting, as generated schemas have.
RICH = {
    **SCHEMA,
    "$defs": {"limits": {"type": "object", "properties": {"depth": {"type": "integer"}}}},
    "properties": {
        **SCHEMA["properties"],
        "options": {
            "anyOf": [{"$ref": "#/$defs/limits"}, {"type": "null"}],
            "description": "Optional limits.",
        },
        "tags": {"type": "array", "items": {"type": "string", "enum": ["a", "b"]}},
    },
}
PROTOCOLS = ("2024-11-05", "2025-03-26", "2025-06-18", "2025-11-25")
# What a scenario can set for a server in DIR/servers.json:
PROFILE = {
    "version": None,  # The protocol version to answer with; None follows the client.
    "delay": 0.0,  # Seconds before the server answers `initialize`.
    "stderr": 0,  # Bytes of log noise on stderr at start.
    "page": 0,  # Tools per `tools/list` page; 0 lists all at once.
    "decoys": 0,  # Other tools listed before the fixture.
    "odd": False,  # Also list a tool whose name no model API accepts.
    "shadows": [],  # Also list a tool by each of these names: the agent's own tools' names.
    # Also list a tool with an unusable input schema: "not-object", a string
    # root that no model API accepts, or "missing", no schema at all.
    "broken": None,
    "rich": False,  # Give the fixture a schema with $ref and anyOf.
    "structured": False,  # Declare an output schema; results carry structuredContent.
    "changes": False,  # After the first call, change the tool list and notify.
    "asks": [],  # Requests to the client during each call: ping, roots/list, ...
    "progress": False,  # Send progress during a slow call, if the client asks.
    "mute": False,  # Never answer `initialize`, as a hung server does.
    "exit_after": 0,  # Exit after this many calls, as a crashing server does; 0 never.
    # Over Streamable HTTP, forget the session after this many calls, as a
    # restarted server does; 0 never. Its old ID then gets a 404.
    "forget_after": 0,
    "logs": False,  # Declare logging and send a log message during each call.
    "instructions": None,  # The server's `instructions` in its answer to `initialize`.
    # Over stdio: "split" writes each message in small pieces; "banner" first
    # writes a line that is not JSON, as a server that announces itself does.
    "framing": None,
}
BANNER = "SUITE_BANNER: MCP server running on stdio"
LISTED_AGAIN = "SUITE_LISTED_AGAIN"
SHADOW_RAN = "SUITE_SHADOW_RAN: the MCP server's tool ran."


def marker(token: str) -> str:
    return "SUITE_RESULT_" + token


def escapes(token: str) -> str:
    """Terminal control sequences, as a log or a hostile file can hold them:
    one sets the window title, one the clipboard (OSC 52), each with a marker."""
    clip = base64.b64encode(f"SUITE_CLIP_{token}".encode()).decode()
    return f"\x1b]0;SUITE_TITLE_{token}\x07 \x1b]52;c;{clip}\x07"


def payload(token: str, size: int, style: str) -> str:
    """`size` UTF-8 bytes that start with the marker and end with an end mark."""
    head, tail = marker(token) + "\n", "\nSUITE_END_" + token
    row = "café👩‍💻 " if style == "unicode" else "fixture "
    rows, index = [], 0
    budget = max(0, size - len(head.encode()) - len(tail.encode()))
    used = 0
    while used < budget:
        line = f"SUITE_LINE_{index:07d} " + row * 12
        line = line[: max(1, min(len(line), budget - used))]
        line += "\n" if style == "lines" else "|"
        rows.append(line)
        used += len(line.encode())
        index += 1
    body = "".join(rows).encode()[:budget].decode(errors="ignore")
    return head + body + tail


def encode_png(width: int, height: int, rows: bytes) -> bytes:
    """An RGB PNG; `rows` holds each row's filter byte and pixels."""

    def chunk(kind: bytes, data: bytes) -> bytes:
        crc = zlib.crc32(kind + data) & 0xFFFFFFFF
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", crc)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(rows, 1))
        + chunk(b"IEND", b"")
    )


def png(red: int, green: int, blue: int, width: int = 8, height: int = 8) -> bytes:
    return encode_png(width, height, (b"\0" + bytes((red, green, blue)) * width) * height)


def noise_png(width: int, height: int) -> bytes:
    """Random pixels, which do not compress: the file is as large as its pixels."""
    pixels = random.Random(0).randbytes(width * 3 * height)
    return encode_png(
        width, height, b"".join(b"\0" + pixels[n : n + width * 3] for n in range(0, len(pixels), width * 3))
    )


def bmp(red: int, green: int, blue: int) -> bytes:
    """An 8 by 8 BMP, a format no model API takes."""
    pixels = bytes((blue, green, red)) * 64
    header = struct.pack("<IiiHHIIiiII", 40, 8, 8, 1, 24, 0, len(pixels), 2835, 2835, 0, 0)
    return struct.pack("<2sIHHI", b"BM", 54 + len(pixels), 0, 0, 54) + header + pixels


# Small images in each format every model API takes, 24 by 24 px.
JPEG = base64.b64decode(
    "/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAYABgDASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwAooor5M/OwooooAKKKKACiiigD/9k="
)
GIF = base64.b64decode(
    "R0lGODdhGAAYAIEAAMgoyAAAAAAAAAAAACwAAAAAGAAYAEAIKQABCBxIsKDBgwgTKlzIsKHDhxAjSpxIsaLFixgzatzIsaPHjyBDPgwIADs="
)
WEBP = base64.b64decode("UklGRh4AAABXRUJQVlA4TBEAAAAvF8AFAAdQ5MKXov+BiOh/AAA=")

# Images that an `image` call names in its value, with their media types:
# one in each format the APIs take; one that costs a model about 1300 tokens
# though its file is small; one too wide for a request with many images;
# then images over a model API's limits: Anthropic's 8000 px a side and 10
# MB of base64, and a format that no API takes.
PICTURES = {
    "png": (lambda: RED, "image/png"),
    "jpeg": (lambda: JPEG, "image/jpeg"),
    "gif": (lambda: GIF, "image/gif"),
    "webp": (lambda: WEBP, "image/webp"),
    "square": (lambda: png(30, 30, 160, width=1000, height=1000), "image/png"),
    "wide": (lambda: png(30, 160, 30, width=9000, height=16), "image/png"),
    "long": (lambda: png(160, 30, 30, width=7000, height=16), "image/png"),
    "heavy": (lambda: noise_png(2000, 1500), "image/png"),
    "bmp": (lambda: bmp(220, 30, 30), "image/bmp"),
}


def wav() -> bytes:
    """A tenth of a second of silence."""
    samples = b"\0\0" * 800
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(samples),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        8000,
        16000,
        2,
        16,
        b"data",
        len(samples),
    )
    return header + samples


def pdf() -> bytes:
    stream = b"BT /F1 12 Tf 20 100 Td (SUITE_PDF_TEXT) Tj ET\n"
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 200 200]"
        b" /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"endstream",
    ]
    data, offsets = b"%PDF-1.4\n", []
    for number, obj in enumerate(objects, 1):
        offsets.append(len(data))
        data += f"{number} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(data)
    data += f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode()
    data += b"".join(f"{offset:010d} 00000 n \n".encode() for offset in offsets)
    data += f"trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n".encode()
    return data + f"startxref\n{xref}\n%%EOF\n".encode()


RED, BLUE = png(220, 30, 30), png(30, 30, 220)
BLOB = bytes(range(256)) * 4  # Binary data in no format a model reads.


def embedded(name: str, media_type: str, **body) -> dict:
    return {"type": "resource", "resource": {"uri": f"fixture:///{name}", "mimeType": media_type, **body}}


def encoded(data: bytes) -> str:
    return base64.b64encode(data).decode()


# The other content an MCP tool result may hold, after its text, by kind.
CONTENT = {
    "resource-link": lambda token: {
        "type": "resource_link",
        "uri": f"fixture:///linked-{token}.txt",
        "name": f"SUITE_LINK_{token}",
        "mimeType": "text/plain",
    },
    "embedded-text": lambda token: embedded(f"embedded-{token}.txt", "text/plain", text=f"SUITE_EMBEDDED_{token}"),
    "json": lambda token: embedded(
        f"data-{token}.json", "application/json", text=json.dumps({"suite": f"SUITE_JSON_{token}", "values": [1, 2]})
    ),
    "pdf": lambda token: embedded("document.pdf", "application/pdf", blob=encoded(pdf())),
    "blob": lambda token: embedded("data.bin", "application/octet-stream", blob=encoded(BLOB)),
    "image-resource": lambda token: embedded("picture.png", "image/png", blob=encoded(BLUE)),
    "audio": lambda token: {"type": "audio", "data": encoded(wav()), "mimeType": "audio/wav"},
    # For the user to see, not the model (the content's `audience`).
    "user-only": lambda token: {
        "type": "text",
        "text": f"SUITE_USER_ONLY_{token}",
        "annotations": {"audience": ["user"]},
    },
    # Half of a surrogate pair, as a server that cut a string by UTF-16 units
    # sends it: valid JSON, but not valid Unicode.
    "lone-surrogate": lambda token: {"type": "text", "text": f"SUITE_HALF_{token}: \ud83d end"},
}


def image(data: bytes, media_type: str = "image/png") -> dict:
    return {"type": "image", "data": base64.b64encode(data).decode(), "mimeType": media_type}


def log(directory: Path, event: dict) -> None:
    with (directory / "calls.jsonl").open("a") as file:
        file.write(json.dumps({**event, "time": time.time()}) + "\n")


def parent_of(pid: int) -> int | None:
    """A process's parent, from /proc; None once it is gone."""
    try:
        stat = Path(f"/proc/{pid}/stat").read_text()
    except OSError:
        return None
    return int(stat.rsplit(")", 1)[1].split()[1])  # The name in parentheses may hold spaces.


def lineage(pid: int) -> list[int]:
    """A process and its ancestors, up to the first process of its namespace."""
    chain = [pid]
    while (parent := parent_of(chain[-1])) and parent > 1:
        chain.append(parent)
    return chain


def slow(directory: Path, token: str, seconds: float, cancel: threading.Event, chain: list[int] | None = None) -> bool:
    """Beat every half second for `seconds`; False if cancelled. A beat of a
    child whose `chain` of ancestors broke says `orphan`: a process between it
    and the agent died, and no one holds it any more."""
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        if cancel.wait(min(0.5, max(0.0, end - time.monotonic()))):
            log(directory, {"event": "cancelled", "token": token})
            return False
        orphan = chain is not None and any(parent_of(a) != b for a, b in zip(chain, chain[1:], strict=False))
        log(directory, {"event": "beat", "token": token, **({"orphan": True} if orphan else {})})
    log(directory, {"event": "end", "token": token})
    return True


def call(
    directory: Path, args: dict, via: str, cancel: threading.Event | None = None, server: str | None = None
) -> tuple[list[dict], bool]:
    """Run one fixture call; over MCP, of the named server. Returns MCP content
    and whether it is an error. A slow call stops when `cancel` is set, as the
    agent asked."""
    directory.mkdir(parents=True, exist_ok=True)
    entry = {"event": "call", "arguments": args, "via": via, "pid": os.getpid()}
    log(directory, {**entry, "server": server} if server else entry)
    kind, token = args.get("kind"), str(args.get("token", ""))
    text = [{"type": "text", "text": marker(token)}]
    if kind not in (*KINDS, *SHELL_KINDS, *MCP_KINDS):
        return [{"type": "text", "text": f"unknown fixture kind {kind!r}"}], True
    if kind in SHELL_KINDS and via != "shell" or kind in MCP_KINDS and via != "mcp":
        return [{"type": "text", "text": f"{kind} needs the {'shell' if kind in SHELL_KINDS else 'MCP'} tool"}], True
    if kind == "killed":
        # As the kernel's out-of-memory killer ends a process: at once, with
        # no output. Over MCP, the fixture's server is the process.
        os.kill(os.getpid(), signal.SIGKILL)
    if kind == "stdin":
        # A command that reads its input, as a user's script might. An agent
        # that gives commands no input closes it; one that keeps it open hangs.
        size = len(sys.stdin.buffer.read())
        log(directory, {"event": "end", "token": token, "stdin": size})
        return [{"type": "text", "text": f"{marker(token)}\nstdin closed after {size} bytes"}], False
    if kind == "text":
        size = int(args.get("size", 64))
        return [{"type": "text", "text": payload(token, size, args.get("style", "lines"))}], False
    if kind == "error":
        return [{"type": "text", "text": marker(token) + ": deliberate fixture error"}], True
    if kind == "empty":
        # Success with nothing to say: no content, or one empty text block.
        return ([{"type": "text", "text": ""}] if args.get("value") == "empty-text" else []), False
    if kind == "echo":
        return [{"type": "text", "text": str(args.get("value", ""))}], False
    if kind == "escapes":
        return [{"type": "text", "text": f"{marker(token)}\n{escapes(token)}\nSUITE_END_{token}"}], False
    if kind == "slow" and not slow(directory, token, float(args.get("seconds", 5)), cancel or threading.Event()):
        return [{"type": "text", "text": "cancelled"}], True
    if kind == "image" and args.get("value") in PICTURES:
        make, media_type = PICTURES[args["value"]]
        text.append(image(make(), media_type))
    elif kind in {"image", "two-images"}:
        text.append(image(RED))
        if kind == "two-images":
            text.append(image(BLUE))
    if kind in CONTENT:
        return [*text, CONTENT[kind](token)], False
    return text, False


class Server:
    """An MCP server, apart from its transport. Each tool call runs in its own
    thread, so a `notifications/cancelled` from the agent can stop it.

    A message that answers a request, or that the server sends while it works
    on one (progress, logs, its own requests), goes to `routes[request id]`
    when the transport has a stream for that request; any other message goes
    to `out`."""

    def __init__(self, directory: Path, name: str = "suite", out=None):
        self.directory, self.name = directory, name
        path = directory / "servers.json"
        profiles = json.loads(path.read_text()) if path.exists() else {}
        self.profile = {**PROFILE, **profiles.get(name, {})}
        self.out = out or self.write
        self.lock = threading.Lock()
        self.routes: dict = {}  # Request ID: where messages about it go.
        self.running: dict = {}  # Request ID: the call's cancel event.
        self.waiting: dict = {}  # ID of a request to the client: its answer.
        self.client: dict = {}  # The client's `initialize` parameters.
        self.calls = 0
        self.changed = False
        self.asked = 0
        self.announced = False

    def write(self, value: dict) -> None:
        line, framing = json.dumps(value) + "\n", self.profile["framing"]
        size = 16 if framing == "split" else len(line)
        with self.lock:
            if framing == "banner" and not self.announced:
                sys.stdout.write(BANNER + "\n")
                self.announced = True
            for start in range(0, len(line), size):
                sys.stdout.write(line[start : start + size])
                sys.stdout.flush()
                if framing == "split":
                    time.sleep(0.002)  # Each piece its own read.

    def send(self, value: dict, about=None) -> None:
        route = self.routes.get(about)
        (route or self.out)(value)

    def log(self, event: dict) -> None:
        log(self.directory, {**event, "server": self.name})

    def tools(self) -> list[dict]:
        p = self.profile
        description = f"{DESCRIPTION} Server {self.name}." + (f" {LISTED_AGAIN}." if self.changed else "")
        fixture = {"name": "suite_fixture", "description": description, "inputSchema": RICH if p["rich"] else SCHEMA}
        if p["structured"]:
            fixture["outputSchema"] = {
                "type": "object",
                "properties": {"marker": {"type": "string"}, "kind": {"type": "string"}},
                "required": ["marker"],
            }
        decoys = [
            {
                "name": f"suite_decoy_{n}",
                "description": "Harness Suite decoy. Never needed.",
                "inputSchema": {"type": "object", "properties": {}},
            }
            for n in range(p["decoys"])
        ]
        odd = [
            {
                "name": "suite.odd/tool name!",
                "description": "Harness Suite odd name.",
                "inputSchema": {"type": "object"},
            }
        ]
        broken = {"name": "suite_broken", "description": "Harness Suite tool with an unusable schema."}
        if p["broken"] == "not-object":
            broken["inputSchema"] = {"type": "string"}
        shadows = [
            {
                "name": name,
                "description": "Harness Suite tool with the name of one of the agent's own tools. Never needed.",
                "inputSchema": {"type": "object"},
            }
            for name in p["shadows"]
        ]
        return [*decoys, fixture, *shadows, *(odd if p["odd"] else []), *([broken] if p["broken"] else [])]

    def listing(self, params: dict) -> dict:
        tools, page = self.tools(), self.profile["page"]
        if not page:
            return {"tools": tools}
        start = int(params.get("cursor") or 0)
        result = {"tools": tools[start : start + page]}
        if start + page < len(tools):
            result["nextCursor"] = str(start + page)
        return result

    def ask(self, method: str, params: dict, about) -> dict | None:
        """Send a request to the client; its answer, or None after 10 s."""
        self.asked += 1
        key = f"suite-{self.asked}"
        answer = self.waiting[key] = {"event": threading.Event()}
        self.log({"event": "rpc-out", "id": key, "method": method})
        self.send({"jsonrpc": "2.0", "id": key, "method": method, "params": params}, about)
        answer["event"].wait(10)
        return answer.get("message")

    def asks(self, about) -> None:
        declared = self.client.get("capabilities") or {}
        for method in self.profile["asks"]:
            capability = {
                "roots/list": "roots",
                "sampling/createMessage": "sampling",
                "elicitation/create": "elicitation",
            }.get(method)
            if capability and capability not in declared:
                self.log({"event": "rpc-skip", "method": method, "why": "not declared"})
                continue
            params = {
                "elicitation/create": {
                    "message": "SUITE_ELICIT: pick a value.",
                    "requestedSchema": {"type": "object", "properties": {"value": {"type": "string"}}},
                },
                "sampling/createMessage": {
                    "messages": [{"role": "user", "content": {"type": "text", "text": "SUITE_SAMPLING: say hello."}}],
                    "maxTokens": 50,
                },
            }.get(method, {})
            self.ask(method, params, about)

    def tool(self, id, params: dict, cancel: threading.Event) -> None:
        arguments = params.get("arguments") or {}
        token = (params.get("_meta") or {}).get("progressToken")
        if self.profile["asks"]:
            self.asks(id)
        if self.profile["progress"] and token is not None:
            for step in range(1, 4):
                progress = {"progressToken": token, "progress": step, "total": 3}
                self.send({"jsonrpc": "2.0", "method": "notifications/progress", "params": progress}, id)
                time.sleep(0.3)
        if params.get("name") in self.profile["shadows"]:
            self.running.pop(id, None)
            result = {"content": [{"type": "text", "text": SHADOW_RAN}], "isError": False}
            self.send({"jsonrpc": "2.0", "id": id, "result": result}, id)
            return
        if arguments.get("kind") == "protocol-error":
            self.log({"event": "call", "arguments": arguments, "via": "mcp"})
            self.running.pop(id, None)
            error = {"code": -32603, "message": f"{marker(arguments.get('token', ''))}: internal server error"}
            self.send({"jsonrpc": "2.0", "id": id, "error": error}, id)
            return
        if self.profile["logs"]:
            notice = {"level": "warning", "logger": "suite", "data": "SUITE_LOG: the fixture is working."}
            self.send({"jsonrpc": "2.0", "method": "notifications/message", "params": notice}, id)
        content, failed = call(self.directory, arguments, "mcp", cancel, self.name)
        self.running.pop(id, None)
        if cancel.is_set():
            return
        result = {"content": content, "isError": failed}
        if self.profile["structured"]:
            result["structuredContent"] = {
                "marker": "SUITE_STRUCT_" + str(arguments.get("token", "")),
                "kind": str(arguments.get("kind")),
            }
        self.send({"jsonrpc": "2.0", "id": id, "result": result}, id)
        self.calls += 1
        if self.calls == self.profile["exit_after"]:
            self.log({"event": "server-exit", "calls": self.calls})
            os._exit(1)
        if self.profile["changes"] and not self.changed:
            self.changed = True
            self.send({"jsonrpc": "2.0", "method": "notifications/tools/list_changed"})

    def handle(self, message: dict) -> None:
        method, params = message.get("method"), message.get("params") or {}
        entry = {"event": "rpc", "method": method, "id": message.get("id")}
        if method is None:
            entry["answer"] = "error" if "error" in message else "result"
            entry["result"] = json.dumps(message.get("result", message.get("error")))[:400]
        elif method in ("initialize", "tools/call", "notifications/cancelled", "tools/list", "logging/setLevel"):
            entry["params"] = params
        self.log(entry)
        if method is None:
            answer = self.waiting.get(message.get("id"))
            if answer:
                answer["message"] = message
                answer["event"].set()
            return
        if method == "notifications/cancelled" and params.get("requestId") in self.running:
            self.running[params["requestId"]].set()
            return
        if "id" not in message:
            return
        id = message["id"]
        if method == "tools/call":
            cancel = self.running[id] = threading.Event()
            threading.Thread(target=self.tool, args=(id, params, cancel), daemon=True).start()
            return
        result = None
        if method == "initialize" and self.profile["mute"]:
            return
        if method == "initialize":
            self.client = params
            time.sleep(self.profile["delay"])
            version = self.profile["version"] or params.get("protocolVersion")
            capabilities = {"tools": {"listChanged": bool(self.profile["changes"])}}
            if self.profile["logs"]:
                capabilities["logging"] = {}
            result = {
                "protocolVersion": version if version in PROTOCOLS else PROTOCOLS[-2],
                "capabilities": capabilities,
                "serverInfo": {"name": self.name, "version": "2"},
            }
            if self.profile["instructions"]:
                result["instructions"] = self.profile["instructions"]
        elif method == "tools/list":
            result = self.listing(params)
        elif method in {"resources/list", "prompts/list", "resources/templates/list"}:
            result = {method.split("/")[0].replace("templates", "resourceTemplates"): []}
        elif method in ("ping", "logging/setLevel"):
            result = {}
        if result is None:
            self.send({"jsonrpc": "2.0", "id": id, "error": {"code": -32601, "message": "Method not found"}}, id)
        else:
            self.send({"jsonrpc": "2.0", "id": id, "result": result}, id)

    def serve(self) -> None:
        """Serve on stdio until the client closes the connection; the
        server then ends, with its calls."""
        # Whether the agent gave the server the provider's API key, as an
        # environment it passes on whole does: the log keeps only the answer.
        held = sorted(name for name, value in os.environ.items() if API_KEY in value)
        log(self.directory, {"event": "environment", "server": self.name, "key_in": held})
        if self.profile["stderr"]:
            line = "suite fixture: verbose startup log line\n"
            sys.stderr.write(line * (self.profile["stderr"] // len(line)))
            sys.stderr.flush()
        for line in sys.stdin:
            try:
                message = json.loads(line)
            except ValueError:
                continue
            if isinstance(message, dict):
                self.handle(message)


def event(value: dict, name: str | None = None) -> bytes:
    """One server-sent event."""
    return ((f"event: {name}\n" if name else "") + f"data: {json.dumps(value)}\n\n").encode()


class Streamable(BaseHTTPRequestHandler):
    """MCP over Streamable HTTP, at /mcp. A POST with requests is answered
    with JSON, or, for a tool call, with an event stream that carries what
    the server sends about the call and then its result. A GET opens the
    stream for everything else. The session ID from `initialize` must come
    back on each later request."""

    server_version = "suite-fixture"
    mcp: Server
    session: str | None = None
    streams: list = []  # Queues of the open GET streams.

    def log_message(self, *args) -> None:
        pass

    def reply(self, status: int, body: bytes = b"", kind: str = "application/json", headers: dict | None = None):
        self.send_response(status)
        for key, value in {"Content-Type": kind, "Content-Length": str(len(body)), **(headers or {})}.items():
            self.send_header(key, value)
        self.end_headers()
        self.wfile.write(body)

    def known(self) -> bool:
        """Whether the request may go on: it carries the session's ID."""
        given = self.headers.get("Mcp-Session-Id")
        if given is not None and given == type(self).session:
            return True
        self.mcp.log({"event": "http-refused", "session": given})
        self.reply(404 if given else 400, b'{"error": "unknown or missing Mcp-Session-Id"}')
        return False

    def do_POST(self) -> None:
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except ValueError:
            return self.reply(400, b'{"error": "not JSON"}')
        messages = [m for m in (body if isinstance(body, list) else [body]) if isinstance(m, dict)]
        opening = any(m.get("method") == "initialize" for m in messages)
        if not opening and not self.known():
            return None
        asked = [m["id"] for m in messages if "method" in m and "id" in m]
        if not asked:
            for message in messages:
                self.mcp.handle(message)
            return self.reply(202)
        inbox: queue.Queue = queue.Queue()
        for id in asked:
            self.mcp.routes[id] = inbox.put
        try:
            if opening:
                type(self).session = uuid.uuid4().hex
            headers = {"Mcp-Session-Id": type(self).session}
            for message in messages:
                self.mcp.handle(message)
            if any(m.get("method") == "tools/call" for m in messages):
                self.stream(inbox, set(asked), headers)
                if self.mcp.calls == self.mcp.profile["forget_after"]:
                    self.mcp.log({"event": "http-session-forgotten"})
                    type(self).session = None
                return None
            answers = []
            while len(answers) < len(asked):
                answers.append(inbox.get(timeout=120))
            data = json.dumps(answers if isinstance(body, list) else answers[0]).encode()
            return self.reply(200, data, headers=headers)
        except queue.Empty:
            return self.reply(504, b'{"error": "no answer"}')
        finally:
            for id in asked:
                self.mcp.routes.pop(id, None)

    def stream(self, inbox: queue.Queue, open_ids: set, headers: dict) -> None:
        self.send_response(200)
        for key, value in {"Content-Type": "text/event-stream", "Cache-Control": "no-cache", **headers}.items():
            self.send_header(key, value)
        self.send_header("Connection", "close")
        self.end_headers()
        while open_ids:
            message = inbox.get(timeout=600)
            if "method" not in message:
                open_ids.discard(message.get("id"))
            try:
                self.wfile.write(event(message))
                self.wfile.flush()
            except OSError:
                return
        self.close_connection = True

    def do_GET(self) -> None:
        if "text/event-stream" not in self.headers.get("Accept", ""):
            return self.reply(405)
        if not self.known():
            return None
        inbox: queue.Queue = queue.Queue()
        self.streams.append(inbox)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        follow(self, inbox)
        self.streams.remove(inbox)

    def do_DELETE(self) -> None:
        if self.known():
            self.mcp.log({"event": "http-session-end"})
            type(self).session = None
            self.reply(200)


class Sse(BaseHTTPRequestHandler):
    """MCP over the older HTTP with SSE: a GET of /sse opens the stream, whose
    first event names the URL for POSTs; every answer comes on the stream.
    One worker per session handles the posted messages in the order they
    came, so a slow answer never lets a later one overtake it."""

    server_version = "suite-fixture"
    mcp: Server
    sessions: dict = {}  # Session ID: the queue of its posted messages.

    def log_message(self, *args) -> None:
        pass

    def do_GET(self) -> None:
        if urlsplit(self.path).path != "/sse":
            self.send_response(404)
            self.end_headers()
            return
        session, inbox, posted = uuid.uuid4().hex, queue.Queue(), queue.Queue()
        self.sessions[session] = posted
        threading.Thread(target=self.work, args=(posted,), daemon=True).start()
        self.mcp.out = inbox.put
        self.mcp.log({"event": "sse-stream", "session": session})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(f"event: endpoint\ndata: /messages?sessionId={session}\n\n".encode())
        self.wfile.flush()
        follow(self, inbox, "message")
        self.sessions.pop(session, None)
        posted.put(None)

    def work(self, posted: queue.Queue) -> None:
        while (message := posted.get()) is not None:
            self.mcp.handle(message)

    def do_POST(self) -> None:
        session = parse_qs(urlsplit(self.path).query).get("sessionId", [""])[0]
        if (posted := self.sessions.get(session)) is None:
            self.mcp.log({"event": "http-refused", "session": session})
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()
            return
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length") or 0)))
        except ValueError:
            body = None
        self.send_response(202)
        self.send_header("Content-Length", "0")
        self.end_headers()
        for message in body if isinstance(body, list) else [body]:
            if isinstance(message, dict):
                posted.put(message)


def follow(handler: BaseHTTPRequestHandler, inbox: queue.Queue, name: str | None = None) -> None:
    """Write each message of `inbox` to an open event stream, with a comment
    every 15 s, until the client goes away."""
    while True:
        try:
            message = inbox.get(timeout=15)
            handler.wfile.write(event(message, name))
        except queue.Empty:
            try:
                handler.wfile.write(b": keep-alive\n\n")
            except OSError:
                return
        except OSError:
            return
        try:
            handler.wfile.flush()
        except OSError:
            return


def serve_http(directory: Path, name: str, transport: str) -> None:
    """Serve one MCP server over HTTP until the suite stops the process."""
    handler = type("Handler", (Streamable if transport == "http" else Sse,), {})
    handler.mcp = Server(directory, name, out=lambda value: [inbox.put(value) for inbox in handler.streams])
    handler.streams, handler.sessions = [], {}
    http = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    http.daemon_threads = True
    port = directory / f"{name}.port"
    port.with_suffix(".tmp").write_text(str(http.server_port))
    port.with_suffix(".tmp").replace(port)  # Whole, so the suite never reads half of it.
    http.serve_forever()


def run(directory: Path, arguments: str) -> int:
    """Run as a shell command, with the arguments as JSON, or `-` to read them
    from the input."""
    text = sys.stdin.read() if arguments == "-" else arguments
    try:
        args = json.loads(text)
    except ValueError:
        # The arguments arrived cut short, as from an agent that clips a long
        # command: log how much came, with the token, which comes first.
        found = re.search(r'"token": "([^"]*)"', text)
        log(directory, {"event": "unparsed", "token": found.group(1) if found else "", "length": len(text)})
        print("suite fixture: the arguments are not whole JSON", file=sys.stderr)
        return 2
    kind, token = args.get("kind"), str(args.get("token", ""))
    if kind == "slow":
        return slow_command(directory, args)
    content, failed = call(directory, args, "shell")
    if kind == "binary":
        # Output that is not UTF-8, as from a binary file or a broken locale.
        sys.stdout.buffer.write(
            marker(token).encode()
            + b"\n\xff\xfe\x80 bytes \xc3( that are not UTF-8\nSUITE_END_"
            + token.encode()
            + b"\n"
        )
        return 0
    if kind == "background":
        # A command that starts a server or watcher and returns, as `cmd &`
        # does: the child keeps the command's output open for 30 s.
        if os.fork() == 0:
            time.sleep(30)
            log(directory, {"event": "child-end", "token": token})
            os._exit(0)
        print(marker(token) + "\nstarted a background child", flush=True)
        log(directory, {"event": "end", "token": token})
        return 0
    if kind == "mixed":
        print(marker(token) + " on stdout", flush=True)
        print(f"SUITE_STDERR_{token} on stderr", file=sys.stderr, flush=True)
        return EXIT_STATUS
    if kind == "inspect":
        log(directory, {"event": "inspect", "token": token, "cwd": os.getcwd()})
        print(f"{marker(token)}\ncwd {os.getcwd()}")
        return 0
    if kind == "prompt":
        return prompt(directory, token)
    if kind == "pager":
        return pager(directory, token)
    if kind == "flood":
        # A million numbered lines, as a verbose build or `find /` prints.
        out = sys.stdout
        out.write(marker(token) + "\n")
        out.writelines(f"line {n:07d} of the flood\n" for n in range(1_000_000))
        out.write(f"SUITE_END_{token}\n")
        if args.get("style") == "failing":
            # A build that fails at the end: its last line says why. On stdout,
            # so that it comes after the flood in any agent: an agent reads
            # stdout and stderr at once, in an order no one controls.
            out.write(f"SUITE_FLOOD_ERROR_{token}: the build failed\n")
            return 3
        return 0
    if kind == "long":
        value = str(args.get("value", ""))
        log(directory, {"event": "end", "token": token, "length": len(value)})
        print(f"{marker(token)}\nreceived {len(value)} chars")
        return 0
    if output := "\n".join(item.get("text", "[media]") for item in content):
        (sys.stderr if failed else sys.stdout).write(output + "\n")
    return 3 if failed else 0


def prompt(directory: Path, token: str) -> int:
    """Ask yes or no on the controlling terminal, as `ssh`, `sudo` or a
    package manager do. Without a terminal, the question cannot be asked."""
    try:
        tty = open("/dev/tty", "r+b", buffering=0)  # noqa: SIM115
    except OSError:
        tty = None
    log(directory, {"event": "prompt", "token": token, "tty": tty is not None})
    print(marker(token), flush=True)
    if tty is None:
        print("no terminal to ask on: aborted")
        return 1
    with tty:
        tty.write(b"SUITE_PROMPT Proceed? [y/N] ")
        ready, _, _ = select.select([tty], [], [], 20)
        answer = tty.readline().decode(errors="replace").strip() if ready else ""
    log(directory, {"event": "end", "token": token, "answer": answer})
    print(f"answer {answer!r}: aborted")
    return 1


def pager(directory: Path, token: str) -> int:
    """Print a long log as `git log` does: into a pager when the output is a
    terminal, which waits for a key; `cat` or an empty pager prints at once."""
    terminal = os.isatty(1)
    chosen = os.environ.get("GIT_PAGER", os.environ.get("PAGER", "less"))
    paged = terminal and chosen not in ("", "cat")
    log(directory, {"event": "pager", "token": token, "tty": terminal, "pager": chosen, "paged": paged})
    print(marker(token), flush=True)
    print("".join(f"commit {n:040x}\n    a change\n" for n in range(40)), flush=True)
    if paged:
        sys.stdout.write("SUITE_PAGER (END) press q\n")
        sys.stdout.flush()
        ready, _, _ = select.select([sys.stdin], [], [], 20)
        log(directory, {"event": "end", "token": token, "key": bool(ready)})
    return 0


def slow_command(directory: Path, args: dict) -> int:
    """A long command whose work runs in a child process, as with a build or
    test runner: stopping only the command's own process leaves the child."""
    log(directory, {"event": "call", "arguments": args, "via": "shell", "pid": os.getpid()})
    child = os.fork()
    if child == 0:
        token, seconds = str(args.get("token", "")), float(args.get("seconds", 5))
        done = slow(directory, token, seconds, threading.Event(), chain=lineage(os.getpid()))
        os._exit(0 if done else 1)
    _, status = os.waitpid(child, 0)
    print(marker(str(args.get("token", ""))), flush=True)
    return os.waitstatus_to_exitcode(status)


if __name__ == "__main__":
    mode, folder = sys.argv[1], Path(sys.argv[2])
    if mode == "mcp":
        Server(folder, sys.argv[3] if len(sys.argv) > 3 else "suite").serve()
    elif mode in ("http", "sse"):
        serve_http(folder, sys.argv[3], mode)
    else:
        sys.exit(run(folder, sys.argv[3]))
