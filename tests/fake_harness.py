"""A tiny agent for testing the suite, with optional defects.

    fake_harness.py --api API --base URL --model M --prompt-file F [--continue] [MCP servers] [defect flags]
    fake_harness.py --tui --api API --base URL --model M [defect flags]

It speaks Chat Completions, Responses or Anthropic Messages (`fake_wire`).

It streams, runs `shell` tool calls in their own process group with an empty
input and a 5 s limit, retries 429/5xx once, and saves its session in $HOME
at the end of a turn so `--continue` can resume it. With
`--mcp-stdio`, `--mcp-http` or `--mcp-sse` it also connects MCP servers and
offers their tools. With `--tui` it is a small
interactive UI: it shows and reads pasted prompts, and `/quit` exits. Each defect flag adds one defect.
"""

import base64
import contextlib
import email.utils
import hashlib
import http.client
import json
import os
import re
import signal
import ssl
import stat
import struct
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
import uuid
import zlib
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import HTTPSHandler, ProxyHandler, Request, build_opener, install_opener, urlopen

import fake_wire as wire

args = sys.argv[1:]
flag = args.__contains__
value = lambda name: args[args.index(name) + 1]  # noqa: E731
if flag("--no-proxy"):
    install_opener(build_opener(ProxyHandler({})))  # Ignores HTTP_PROXY and its kin.
session = Path(os.environ["HOME"]) / "session.json"
CACHE_KEY = "suite-" + hashlib.sha256(str(session).encode()).hexdigest()[:16]
SHELL = {
    "type": "function",
    "function": {
        "name": "shell",
        "description": "Run a shell command",
        "parameters": {"type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]},
    },
}
READ = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file",
        "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    },
}
WRITE = {
    "type": "function",
    "function": {
        "name": "write",
        "description": "Write a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
            "required": ["path", "content"],
        },
    },
}
EDIT = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": "Replace text that occurs once in a file",
        "parameters": {
            "type": "object",
            "properties": {"path": {"type": "string"}, "old": {"type": "string"}, "new": {"type": "string"}},
            "required": ["path", "old", "new"],
        },
    },
}
AGENT = {
    "type": "function",
    "function": {
        "name": "agent",
        "description": "Hand a task to a sub-agent, which has the same tools; its answer",
        "parameters": {"type": "object", "properties": {"prompt": {"type": "string"}}, "required": ["prompt"]},
    },
}
NAME = re.compile(r"[a-zA-Z0-9_-]{1,64}")


class Mcp:
    """An MCP client for one server, apart from its transport, restarted if
    it dies. Each server's tools are offered as `mcp__NAME__TOOL`."""

    def __init__(self, name: str):
        self.name = name
        self.lock = threading.Lock()
        self.pending: dict = {}
        self.number = 0
        self.tools: list[dict] = []
        self.changed = False
        self.started = False

    # What each transport does.

    def open(self) -> None:
        raise NotImplementedError

    def send(self, value: dict) -> None:
        raise NotImplementedError

    def alive(self) -> bool:
        raise NotImplementedError

    # The protocol.

    def start(self) -> None:
        self.started = True
        self.open()
        capabilities = {} if flag("--mcp-no-capabilities") else {"roots": {}, "elicitation": {}}
        answer = self.request(
            "initialize",
            {
                "protocolVersion": "2025-06-18",
                "capabilities": capabilities,
                "clientInfo": {"name": "fake", "version": "1"},
            },
            timeout=15,
        )
        if "error" in answer:
            self.tools = []  # Go on without the server.
            return
        if flag("--mcp-list-early"):
            self.load()
        self.notify("notifications/initialized", {})
        self.load()

    def notify(self, method: str, params: dict) -> None:
        self.send({"jsonrpc": "2.0", "method": method, "params": params})

    def request(self, method: str, params: dict, timeout: float | None = None) -> dict:
        """The answer, or {"error": ...} if the server failed, died or timed out."""
        with self.lock:
            self.number += 1
            number, done = self.number, threading.Event()
        self.pending[number] = {"done": done}
        self.send({"jsonrpc": "2.0", "id": number, "method": method, "params": params})
        deadline = time.monotonic() + timeout if timeout else None
        while not done.wait(0.05):
            if not self.alive():
                return {"error": {"message": "MCP server exited"}}
            if deadline and time.monotonic() > deadline:
                return {"error": {"message": f"MCP server did not answer {method}"}}
        return self.pending.pop(number)["message"]

    def receive(self, message: dict) -> None:
        method = message.get("method")
        if method is None:
            if waiting := self.pending.get(message.get("id")):
                waiting["message"] = message
                waiting["done"].set()
        elif method == "notifications/tools/list_changed":
            self.changed = True
        elif "id" in message:
            result = {
                "ping": {},
                "roots/list": {"roots": [{"uri": f"file://{os.getcwd()}", "name": "workspace"}]},
                "elicitation/create": {"action": "decline"},
            }.get(method)
            if result is None or flag("--mcp-mute"):
                return
            self.send({"jsonrpc": "2.0", "id": message["id"], "result": result})

    def load(self) -> None:
        self.tools, cursor = [], None
        while True:
            answer = self.request("tools/list", {"cursor": cursor} if cursor else {})
            self.tools += answer["result"]["tools"]
            cursor = answer["result"].get("nextCursor")
            if not cursor or flag("--mcp-first-page"):
                break
        self.changed = False

    def prefix(self) -> str:
        if flag("--mcp-bare-names"):
            return ""
        return "mcp__" if flag("--mcp-flat-names") else f"mcp__{self.name}__"

    def offered(self) -> list[dict]:
        if not self.started or not self.alive():
            self.start()
        if self.changed:
            self.load()
        found = []
        for tool in self.tools:
            name = self.prefix() + tool["name"]
            schema = tool.get("inputSchema")
            # The model APIs take only valid names and object schemas: skip the rest.
            usable = NAME.fullmatch(name) and isinstance(schema, dict) and schema.get("type") == "object"
            if usable or flag("--mcp-keep-odd"):
                schema = {"type": "object"} if flag("--mcp-flat-schema") else schema
                found.append(
                    {
                        "type": "function",
                        "function": {"name": name, "description": tool["description"], "parameters": schema},
                    }
                )
        return found

    def call(self, name: str, arguments: dict, images: list) -> str:
        params = {"name": name.removeprefix(self.prefix()), "arguments": arguments, "_meta": {"progressToken": 1}}
        answer = self.request("tools/call", params)
        if "error" in answer:
            return "Error: " + answer["error"]["message"]
        result = answer["result"]
        parts = []
        for item in result.get("content", []):
            if item["type"] == "text":
                parts.append(item["text"])
            elif item["type"] == "resource_link":
                parts.append(f"[link {item['uri']}]")
            elif item["type"] == "resource" and "text" in item["resource"]:
                parts.append(item["resource"]["text"])
            elif item["type"] == "image":
                images.append((base64.b64decode(item["data"]), item["mimeType"]))
                parts.append("[image]")
            elif flag("--raw-content"):
                parts.append(json.dumps(item))  # Binary content as its base64 text.
            else:
                parts.append(f"[{item['type']}]")
        text = "\n".join(parts)
        if not flag("--keep-surrogates"):
            # Half of a surrogate pair is not Unicode: it becomes U+FFFD.
            text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
        return ("Error: " + text) if result.get("isError") else text


class Stdio(Mcp):
    def __init__(self, name: str, command: list[str]):
        super().__init__(name)
        self.command, self.process = command, None

    def open(self) -> None:
        self.process = subprocess.Popen(
            self.command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        threading.Thread(target=self.read, args=(self.process,), daemon=True).start()

    def send(self, value: dict) -> None:
        with self.lock, contextlib.suppress(BrokenPipeError):
            self.process.stdin.write((json.dumps(value) + "\n").encode())
            self.process.stdin.flush()

    def alive(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def read(self, process) -> None:
        """Each message, one per line; a line that is not JSON is skipped,
        or with --mcp-strict-lines ends the reader."""
        for line in process.stdout:
            try:
                message = json.loads(line)
            except ValueError:
                if flag("--mcp-strict-lines"):
                    return
                continue
            self.receive(message)


def events(response):
    """The data of each server-sent event, as (event, data)."""
    kind, data = "message", []
    for raw in response:
        line = raw.decode().rstrip("\r\n")
        if line.startswith("event:"):
            kind = line[6:].strip()
        elif line.startswith("data:"):
            data.append(line[5:].strip())
        elif not line and data:
            yield kind, "\n".join(data)
            kind, data = "message", []


class Streamable(Mcp):
    """Streamable HTTP: each message is a POST; the answer comes back as JSON
    or as an event stream."""

    def __init__(self, name: str, url: str):
        super().__init__(name)
        self.url, self.session_id, self.up = url, None, True

    def open(self) -> None:
        self.session_id, self.up = None, True

    def send(self, value: dict) -> None:
        threading.Thread(target=self.post, args=(value,), daemon=True).start()

    def post(self, value: dict) -> None:
        headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
        if self.session_id:
            headers["Mcp-Session-Id"] = self.session_id
        try:
            with urlopen(Request(self.url, json.dumps(value).encode(), headers), timeout=60) as response:
                self.session_id = response.headers.get("Mcp-Session-Id") or self.session_id
                if response.headers.get_content_type() == "text/event-stream":
                    for _, data in events(response):
                        self.receive(json.loads(data))
                elif body := response.read():
                    self.receive(json.loads(body))
        except HTTPError as error:
            if error.code == 404 and self.session_id and not flag("--mcp-keep-session"):
                # The server forgot the session: open a new one, then send again.
                self.session_id = None
                self.start()
                self.post(value)
                return
            self.up = False
        except (URLError, OSError):
            self.up = False

    def alive(self) -> bool:
        return self.up


class Sse(Mcp):
    """The older HTTP with server-sent events: one GET stream brings every
    message from the server, after an `endpoint` event that says where to
    POST."""

    def __init__(self, name: str, url: str):
        super().__init__(name)
        self.url, self.endpoint, self.up = url, None, False

    def open(self) -> None:
        found = threading.Event()
        threading.Thread(target=self.read, args=(found,), daemon=True).start()
        found.wait(10)

    def read(self, found: threading.Event) -> None:
        try:
            with urlopen(Request(self.url, headers={"Accept": "text/event-stream"}), timeout=None) as response:
                self.up = True
                for kind, data in events(response):
                    if kind == "endpoint":
                        parts = urlsplit(self.url)
                        self.endpoint = data if "://" in data else f"{parts.scheme}://{parts.netloc}{data}"
                        found.set()
                    else:
                        self.receive(json.loads(data))
        except (HTTPError, URLError, OSError):
            pass
        self.up = False
        found.set()

    def send(self, value: dict) -> None:
        headers = {"Content-Type": "application/json"}
        with contextlib.suppress(HTTPError, URLError, OSError, ValueError):
            urlopen(Request(self.endpoint, json.dumps(value).encode(), headers), timeout=30).close()

    def alive(self) -> bool:
        return self.up


def servers() -> list[Mcp]:
    """The MCP servers of `--mcp-stdio NAME PY FIXTURE DIR`, `--mcp-http NAME
    URL` and `--mcp-sse NAME URL`, in order; also those of the workspace's
    `.fake-mcp.json` once the user trusts it (`--workspace-trusted`), or with
    the defect `--trust-workspace`, at once."""
    found = []
    for index, word in enumerate(args):
        rest = args[index + 1 :]
        if word == "--mcp-stdio":
            name, python, fixture, folder = rest[:4]
            found.append(Stdio(name, [python, fixture, "mcp", folder, name]))
        elif word == "--mcp-http":
            found.append(Streamable(*rest[:2]))
        elif word == "--mcp-sse":
            found.append(Sse(*rest[:2]))
    project = Path(".fake-mcp.json")
    if (flag("--workspace-trusted") or flag("--trust-workspace")) and project.exists():
        found += [Stdio(name, command) for name, command in json.loads(project.read_text())["servers"].items()]
    return found


mcp = servers()
late = flag("--mcp-late")  # The first request goes out before the MCP tools are listed.


def mcp_server(tool: str) -> Mcp | None:
    """The server whose tool the model called: the first, if names collide."""
    return next((server for server in mcp if tool.startswith(server.prefix())), None)


def lines(response):
    """The lines of a response body, gzip or not, as they arrive."""
    unpack = zlib.decompressobj(wbits=31) if response.headers.get("Content-Encoding") == "gzip" else None
    buffer = b""
    while chunk := response.read1(65536):
        buffer += unpack.decompress(chunk) if unpack else chunk
        *done, buffer = buffer.split(b"\n")
        yield from done
    if buffer:
        yield buffer


def base() -> str:
    """The configured base URL; with --drop-path, the host's own /v1 instead."""
    url = urlsplit(value("--base"))
    return f"{url.scheme}://{url.netloc}/v1" if flag("--drop-path") else value("--base")


def asked_wait(headers) -> float:
    """Seconds the provider asked to wait: `retry-after-ms`, or `retry-after`
    in seconds or as an HTTP date; a short pause if it asked nothing."""
    if ms := headers.get("retry-after-ms"):
        return float(ms) / 1000
    if after := headers.get("retry-after"):
        try:
            return float(after)
        except ValueError:
            return max(0.0, email.utils.parsedate_to_datetime(after).timestamp() - time.time())
    return 0.2


def post(messages: list, tools: list, options: dict) -> wire.Turn:
    """Stream one reply. Retry one transient error."""
    api = value("--api")
    path, headers, body = wire.request(api, model(), messages, tools, options, not flag("--no-cache-marks"))
    if flag("--no-key"):
        headers = {}  # Forgets the configured key.
    headers = {**headers, "Content-Type": "application/json", "Accept-Encoding": "gzip"}
    if flag("--header"):
        name, _, given = value("--header").partition(":")
        headers[name.strip()] = given.strip()
    for attempt in range(2):
        try:
            request = Request(base() + path, json.dumps(body).encode(), headers)
            with provider_opener().open(request, timeout=30) as response:
                return wire.read(api, sse(response))
        except HTTPError as error:
            error.body = error.read().decode(errors="replace")
            wait = asked_wait(error.headers)
            if error.code in (429, 500, 502, 503, 529) and attempt == 0 and not flag("--no-retry") and wait <= 60:
                time.sleep(wait)
                continue
            raise
        except (URLError, ConnectionError, http.client.HTTPException):  # The last: a body cut short.
            if attempt == 0 and not flag("--no-retry"):
                time.sleep(0.2)
                continue
            raise
    raise RuntimeError("unreachable")


def tls() -> ssl.SSLContext:
    """How the agent checks the provider's certificate: with the CAs the
    environment names, as Python's default does."""
    if flag("--insecure"):
        return ssl._create_unverified_context()
    if flag("--ignore-ca-env"):
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
        context.load_verify_locations(capath="/etc/ssl/certs")  # The system's CAs only.
        return context
    return ssl.create_default_context()


def provider_opener():
    """How the agent reaches the provider: with its TLS checks, and through
    the proxy the environment names unless --no-proxy."""
    handlers = [HTTPSHandler(context=tls())]
    if flag("--no-proxy"):
        handlers.append(ProxyHandler({}))
    return build_opener(*handlers)


def sse(response):
    """The data of each server-sent event, as it arrives."""
    data = []
    for raw in lines(response):
        line = raw.decode().rstrip("\r")
        if flag("--short-lines") and len(raw) > 65536:
            raise RuntimeError("token too long")  # As a default line scanner fails.
        if line.startswith("data:"):
            data.append(line[5:].removeprefix(" "))
        elif not line and data:
            yield "\n".join(data)
            data = []


def model() -> str:
    return "wrong-model" if flag("--wrong-model") else value("--model")


KEPT = 50_000  # Characters of a command's output that the model gets: the start and the end.


def execute(command: str) -> str:
    """Run a command; its output, with its exit status if it failed. The
    result comes when the command exits, even if a child keeps its output open."""
    stdin = subprocess.PIPE if flag("--stdin-pipe") else subprocess.DEVNULL
    # A script file holds a command of any length, which one argument of
    # `sh -c` cannot. Its own session makes the command a process group,
    # without a terminal, unless --share-terminal.
    script = tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False)  # noqa: SIM115
    with script:
        script.write(command)
    # With --pty, the output is a terminal, as some agents give commands.
    terminal = os.openpty() if flag("--pty") else None
    process = subprocess.Popen(
        ["sh", script.name],
        stdin=stdin,
        stdout=terminal[1] if terminal else subprocess.PIPE,
        stderr=subprocess.STDOUT,
        start_new_session=not flag("--share-terminal"),
    )
    if terminal:
        os.close(terminal[1])

    def kill():
        if flag("--kill-parent-only"):
            process.kill()  # Its children run on.
        else:
            os.killpg(process.pid, signal.SIGKILL)

    chunks: list[bytes] = []

    def read():
        if terminal:
            with contextlib.suppress(OSError):  # The terminal closes with the command.
                while data := os.read(terminal[0], 65536):
                    chunks.append(data)
            return
        while data := process.stdout.read1(65536):
            chunks.append(data)

    reader = threading.Thread(target=read, daemon=True)
    reader.start()
    deadline = time.monotonic() + 5
    try:
        while process.poll() is None:
            if time.monotonic() > deadline:
                kill()
                return "timed out after 5 s"
            time.sleep(0.05)
    finally:
        os.unlink(script.name)
    reader.join(timeout=60 if flag("--wait-eof") else 0.5)
    data = b"".join(chunks)
    output = data.decode() if flag("--strict-decode") else data.decode(errors="replace")
    if len(output) > KEPT and flag("--head-only"):
        output = output[:KEPT]
    elif len(output) > KEPT and not flag("--no-clip"):
        output = f"{output[: KEPT // 2]}\n[{len(output) - KEPT} characters left out]\n{output[-KEPT // 2 :]}"
    if process.returncode == 0 or flag("--hide-status"):
        return output
    return f"{output}\n[exit status {process.returncode}]"


# What each read saw, for --edit-stale, which edits that copy.
seen: dict[str, str] = {}


def read(path: str) -> str:
    """A file's text, at most 50 KB of it; an error if it is missing or not a
    regular file, as a directory, a pipe or a device is."""
    try:
        if not stat.S_ISREG(os.stat(path).st_mode) and not flag("--read-special"):
            return "Error: not a regular file"
        with open(path, "rb") as file:
            data = file.read() if flag("--read-whole") else file.read(51200)
    except OSError as error:
        return f"Error: {error.strerror}"
    seen[path] = data.decode(errors="replace")
    return seen[path]


MAGIC = {b"\x89PNG": "image/png", b"\xff\xd8": "image/jpeg", b"GIF8": "image/gif", b"BM": "image/bmp"}


def picture(path: str) -> tuple[bytes, str] | None:
    """An image file's bytes and media type, by its first bytes. Only a
    regular file is opened: opening a pipe would block."""
    try:
        if not stat.S_ISREG(os.stat(path).st_mode):
            return None
        with open(path, "rb") as file:
            data = file.read(16 * 1024**2)
    except OSError:
        return None
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return data, "image/webp"
    return next(((data, kind) for magic, kind in MAGIC.items() if data.startswith(magic)), None)


def refusal(data: bytes, media_type: str) -> str | None:
    """Why the APIs would refuse the image: the strictest rules, Anthropic's."""
    if flag("--no-images"):
        return "this agent sends no images"
    if flag("--send-any-image"):
        return None
    if media_type not in ("image/png", "image/jpeg", "image/gif", "image/webp"):
        return f"{media_type} is not a format the API takes"
    if len(data) > 7.5 * 1024**2:
        return "over 10 MB of base64"
    if side(data, media_type) > 8000:
        return "over 8000 px a side"
    return None


def side(data: bytes, media_type: str) -> int:
    """The longer side of a PNG in pixels; 0 for another format."""
    return max(struct.unpack(">II", data[16:24])) if media_type == "image/png" else 0


def attach(images: list, messages: list) -> None:
    """Images from tool results, in a user message after the results: Chat
    tool messages hold only text. What the API would refuse is left out."""
    parts = []
    for data, media_type in images:
        if why := refusal(data, media_type):
            parts.append({"type": "text", "text": f"An image was left out: {why}."})
        else:
            url = f"data:{media_type};base64,{base64.b64encode(data).decode()}"
            parts.append({"type": "image_url", "image_url": {"url": url}})
    if parts:
        messages.append({"role": "user", "content": [{"type": "text", "text": "Images from the tools:"}, *parts]})
    if value("--api") == "anthropic" and not flag("--keep-all-images"):
        forget_images(messages, 20, 3000)


def decoded(url: str) -> tuple[bytes, str]:
    """The bytes and media type of a `data:` URL."""
    head, data = url.removeprefix("data:").split(";base64,", 1)
    return base64.b64decode(data), head


def forget_images(messages: list, kept: int, limit: int) -> None:
    """Leave out the oldest images, so the conversation holds at most `kept`
    when one is over `limit` px a side: Anthropic's limit on a request with
    many images."""
    holding = [m for m in messages if m["role"] == "user" and isinstance(m["content"], list)]
    urls = [p["image_url"]["url"] for m in holding for p in m["content"] if p["type"] == "image_url"]
    count = len(urls)
    if count <= kept or all(side(*decoded(url)) <= limit for url in urls):
        return
    for message in holding:
        for n, part in enumerate(message["content"]):
            if count <= kept:
                return
            if part["type"] == "image_url":
                message["content"][n] = {"type": "text", "text": "[an earlier image, left out]"}
                count -= 1


def write(path: str, content: str) -> str:
    text = content.replace("\r\n", "\n") if flag("--text-mode-write") else content
    try:
        if not flag("--no-mkdir"):
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as file:
            file.write(text)
    except OSError as error:
        return f"Error: {error.strerror}"
    return f"Wrote {len(text)} chars to {path}"


def edit(path: str, old: str, new: str) -> str:
    """Replace `old` where it occurs once; an error otherwise. The file keeps
    its line ends: the model's new lines become CRLF in a CRLF file."""
    ends = None if flag("--edit-lf") else ""  # None reads and writes the platform's new lines.
    try:
        if flag("--edit-stale") and path in seen:
            text = seen[path]
        else:
            with open(path, newline=ends) as file:
                text = file.read()
    except OSError as error:
        return f"Error: {error.strerror}"
    if "\r\n" in text and old not in text:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    count = text.count(old)
    if count == 0 or count > 1 and not flag("--edit-first"):
        return f"Error: the old text occurs {count} times in {path}; it must occur once"
    with open(path, "w", newline=ends) as file:
        file.write(text.replace(old, new, 1))
    return f"Edited {path}"


MAX_TOOLS = 128  # Chat Completions takes no more.
MCP_CALL = {
    "type": "function",
    "function": {
        "name": "mcp_call",
        "description": "Call one of the MCP tools that the system prompt names.",
        "parameters": {
            "type": "object",
            "properties": {"tool": {"type": "string"}, "arguments": {"type": "object"}},
            "required": ["tool"],
        },
    },
}


def run_tools(calls: dict, messages: list) -> None:
    """Run the calls, and add their results."""
    images: list[tuple[bytes, str]] = []
    for call in calls.values():
        try:
            arguments = json.loads(call["arguments"])
            if call["name"] in ("shell", "read"):
                arguments = arguments["command" if call["name"] == "shell" else "path"]
        except (ValueError, KeyError, TypeError):
            if not flag("--run-broken"):
                messages.append({"role": "tool", "tool_call_id": call["id"], "content": "invalid arguments"})
                continue
            raw = call["arguments"].split('"command": "', 1)[-1].rstrip('"}')
            try:
                arguments = json.loads(f'"{raw}"')
            except ValueError:
                arguments = raw
        if call["name"] in ("shell", "write", "edit") and flag("--read-only") and not flag("--read-only-ignored"):
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": "Error: read-only mode"})
            continue
        for _ in range(2 if flag("--run-twice") else 1):
            if call["name"] == "shell":
                output = execute(arguments)
            elif call["name"] == "read" and flag("--image-read-error") and picture(arguments):
                output = "tool error: images need the terminal UI"
            elif call["name"] == "read" and (found := picture(arguments)):
                images.append(found)
                output = "[image]"
            elif call["name"] == "read":
                output = read(arguments)
            elif call["name"] == "write":
                output = write(arguments["path"], arguments["content"])
            elif call["name"] == "edit":
                output = edit(arguments["path"], arguments["old"], arguments["new"])
            elif call["name"] == "agent":
                output = subagent(arguments["prompt"], messages)
            elif call["name"] == "mcp_call" and (server := mcp_server(arguments.get("tool", ""))):
                output = server.call(arguments["tool"], arguments.get("arguments") or {}, images)
            elif server := mcp_server(call["name"]):
                output = server.call(call["name"], arguments, images)
            else:
                output = f"Error: no tool named {call['name']}"
        if flag("--clip"):
            output = output[:100]
        if flag("--raw-results") and flag("--tui"):
            sys.stdout.write(output.replace("\n", "\r\n") + "\r\n")  # Tool output to the terminal as it came.
            sys.stdout.flush()
        if not flag("--drop-result"):
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": output})
    attach(images, messages)


background: list[threading.Thread] = []  # Sub-agents still at work.
reports: list[str] = []  # Their answers, for the main loop's next turn.


def subagent(task: str, history: list) -> str:
    """Run a sub-agent's turn on its own conversation, or with --subagent-forks
    on a copy of this one; its answer, or with --subagent-lost a bare notice,
    or an error result if its provider fails.
    With --subagent-background it runs on, and its answer comes as a message
    once the main turn ends."""
    calling = max(n for n, m in enumerate(history) if m.get("tool_calls"))
    messages = [*(history[:calling] if flag("--subagent-forks") else []), {"role": "user", "content": task}]
    if flag("--subagent-background"):
        worker = threading.Thread(target=lambda: reports.append(turn(messages)), daemon=True)
        worker.start()
        background.append(worker)
        return "The sub-agent is working in the background; its answer comes as a message."
    try:
        answer = turn(messages)
    except (HTTPError, URLError, ConnectionError) as error:
        if flag("--subagent-crash"):
            raise  # The sub-agent's error ends the whole agent.
        answer = f"Error: the sub-agent failed: {error} {getattr(error, 'body', '')}".strip()
    return "The sub-agent finished." if flag("--subagent-lost") else answer


def reported(messages: list) -> bool:
    """Wait for the sub-agents at work; whether their answers came, as a message."""
    while background:
        background.pop().join()
    if not reports:
        return False
    messages.append({"role": "user", "content": "\n".join(f"<subagent>{r}</subagent>" for r in reports)})
    reports.clear()
    return True


def clear(messages: list) -> bool:
    """Compact: clear the tool results and images before the last few
    messages. Whether anything was left to clear."""
    cleared = False
    if flag("--forget-rules") and messages and messages[0]["role"] == "system":
        del messages[0]  # As an agent whose summary replaces the instructions too.
    for message in messages[:-4]:
        if message["role"] == "tool" and message["content"] != "[cleared]":
            message["content"], cleared = "[cleared]", True
        elif message["role"] == "user" and isinstance(message["content"], list):
            message["content"], cleared = "Images from the tools: [cleared]", True
    return cleared


def turn(messages: list) -> str:
    """Run the model and its tools until it answers. Raise on a provider error."""
    global late
    text = ""
    for _ in range(100):
        sent = messages
        if flag("--volatile-prefix"):
            sent = [{"role": "system", "content": f"Request at {time.time()}"}, *messages]
        own = [SHELL, READ, WRITE, EDIT, AGENT]
        served = [tool for server in ([] if late else mcp) for tool in server.offered()]
        tools = [*own, *served]
        if value("--api") == "chat" and len(tools) > MAX_TOOLS and not flag("--all-tools"):
            # Over Chat's limit, the MCP tools go behind one tool, named in the system prompt.
            names = ", ".join(tool["function"]["name"] for tool in served)
            tools = [*own, MCP_CALL]
            sent = [{"role": "system", "content": f"MCP tools, called through mcp_call: {names}"}, *sent]
        options = {"temperature": float(value("--temperature"))} if flag("--temperature") else {}
        if value("--api") != "anthropic":
            # One key per session, kept across launches, unless a defect makes one per request.
            options["prompt_cache_key"] = uuid.uuid4().hex if flag("--cache-key-per-request") else CACHE_KEY
        try:
            reply = post(sent, tools, options)
            if late:
                late = False
                for server in mcp:
                    server.offered()
        except HTTPError as error:
            if any(w in error.body for w in wire.OVERFLOW) and not flag("--no-compact") and clear(messages):
                continue
            raise
        text, calls = reply.text, reply.calls
        if flag("--calls-need-finish") and reply.stop == "stop":
            calls = {}  # Runs calls only when the finish reason says `tool_calls`.
        if not calls and (reply.early or not text) and not flag("--trust-early-end"):
            raise ValueError("the reply ended early, or said nothing")
        if not flag("--keep-ids"):
            unique(calls, messages)
        assistant = {"role": "assistant", "content": text or None}
        if not flag("--drop-reasoning"):
            if reply.reasoning:
                assistant["reasoning_content"] = reply.reasoning
            if flag("--strip-signatures"):
                for item in reply.items:
                    item.pop("signature", None)
                    item.pop("encrypted_content", None)
            if reply.items:
                assistant["reasoning_items"] = {"api": value("--api"), "items": reply.items}
        if calls:
            assistant["tool_calls"] = [
                {"id": c["id"], "type": "function", "function": {"name": c["name"], "arguments": c["arguments"]}}
                for c in calls.values()
            ]
        messages.append(assistant)
        if flag("--save-early"):
            save(messages)  # Before the tools run: a crash leaves calls without results.
        if not calls:
            break
        run_tools(calls, messages)
    return text


def unique(calls: dict, messages: list) -> None:
    """Give a call whose ID the model used before an ID of its own: some
    providers reuse `call_0` on every turn, and some APIs reject that."""
    used = {c["id"] for m in messages for c in m.get("tool_calls") or []}
    for call in calls.values():
        while call["id"] in used:
            call["id"] += "_again"
        used.add(call["id"])


def save(messages: list) -> None:
    if flag("--reasoning-as-user"):
        saved = []
        for message in messages:
            message.pop("reasoning_items", None)
            if message.get("reasoning_content"):
                saved.append({"role": "user", "content": "Your earlier reasoning: " + message.pop("reasoning_content")})
            saved.append(message)
        messages = saved
    session.write_text(json.dumps(messages))


def start() -> list:
    messages = json.loads(session.read_text()) if flag("--continue") and session.exists() else []
    rules = Path("AGENTS.md")
    if not messages and rules.exists():
        messages.append({"role": "system", "content": rules.read_text()})
    if not messages and flag("--system"):
        role = "user" if flag("--system-as-user") else "system"
        messages.append({"role": role, "content": value("--system")})
    return messages


def says(error: Exception) -> str:
    """What the agent tells the user about a failed request."""
    return f"provider error: {error}"


def headless() -> int:
    if flag("--log-to-stdout"):
        print("fake: starting", flush=True)
    messages = start()
    with open(value("--prompt-file"), newline="") as file:  # The prompt as written, CRLF and all.
        prompt = file.read()
    if flag("--mangle-input"):
        prompt = prompt.replace("\t", "    ")  # Tabs become spaces, as some editors do.
    if not prompt.strip() and not flag("--allow-empty-prompt"):
        print("fake: the prompt is empty", file=sys.stderr)
        return 2
    messages.append({"role": "user", "content": prompt})
    try:
        text = turn(messages)
        while reported(messages):
            text = turn(messages)
    except (HTTPError, URLError, ValueError) as error:
        print(says(error), file=sys.stderr)
        emit({"type": "error", "message": says(error)})
        return 0 if flag("--exit-zero") else 1
    save(messages)
    if flag("--cut-output"):
        output = json.dumps({"type": "result", "text": text}) if flag("--json") else text
        sys.stdout.write(output[:4096])
        sys.stdout.flush()
        os._exit(0)  # Exits before the rest of its output is written.
    emit({"type": "result", "text": text})
    return 0


def emit(event: dict) -> None:
    """Headless output: the answer as text, or with `--json` one JSON event
    per line."""
    if flag("--color"):
        print("\x1b[2K\x1b[32m", end="")  # A spinner's leftovers, as when the agent thinks stdout is a terminal.
    if flag("--json"):
        print(json.dumps(event), flush=True)
    elif event["type"] == "result":
        print(event["text"], flush=True)


def interactive() -> int:
    """Read keys: a bracketed paste is the prompt, and Enter sends it. A
    prompt sent while a turn runs is lost."""
    show = lambda text: (sys.stdout.write(text.replace("\n", "\r\n")), sys.stdout.flush())  # noqa: E731
    messages, draft = start(), ""
    fd = sys.stdin.fileno()
    saved = termios.tcgetattr(fd)
    tty.setraw(fd)
    sys.stdout.write("\x1b[?2004h")  # Ask for bracketed paste.

    running = threading.Event()  # Set while a turn runs.

    def work(prompt: str) -> None:
        try:
            messages.append({"role": "user", "content": prompt})
            show(turn(messages) + "\n> ")
        except HTTPError as error:
            show(f"provider error {error.code}: {error.body}\n> ")
        except (URLError, ValueError) as error:
            show(f"{says(error)}\n> ")
        finally:
            running.clear()
        save(messages)

    try:
        show("fake ready\n> ")
        ready = time.monotonic() + (3 if flag("--slow-input") else 0)
        lost = pasting = typed = False
        while True:
            data = os.read(fd, 65536).decode(errors="replace")
            if time.monotonic() < ready:
                continue  # Still starting: input is lost.
            rest = data
            # A long paste comes in several reads; a line break in it is no Enter.
            if pasting or "\x1b[200~" in data:
                inside = data if pasting else data.split("\x1b[200~", 1)[1]
                pasted, end, rest = inside.partition("\x1b[201~")
                pasting = not end
                draft += pasted.replace("\r", "\n")
                show("[pasted text]" if flag("--collapse-paste") else pasted.replace("\r", "\n"))
                if pasting:
                    continue
            typed = typed or bool(rest.rstrip("\r"))
            if not rest.endswith("\r"):
                draft += rest
                continue
            if flag("--paste-needs-typing") and not typed:
                continue  # A paste alone is an attachment that Enter does not send.
            prompt, draft, typed = draft + rest[:-1], "", False
            if flag("--paste-limit"):
                prompt = prompt[:16384]  # As a TUI with a fixed input buffer.
            if prompt.strip() == "/quit":
                return 0
            if flag("--lose-first-prompt") and not lost:
                lost = True
                # Erases the prompt where it showed, and sends nothing.
                sys.stdout.write("\x1b[1A\x1b[2K" * prompt.count("\n") + "\r\x1b[2K> ")
                sys.stdout.flush()
                continue
            if running.is_set():
                continue
            running.set()
            threading.Thread(target=work, args=(prompt,), daemon=True).start()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, saved)


def unknown_option() -> str | None:
    """An option the fake does not have: every option it has is named in its
    source or its wire module's. The words after a value option are values,
    not options."""
    source = Path(__file__).read_text() + Path(wire.__file__).read_text()
    known = set(re.findall(r'"(--[a-z][\w-]*)"', source))
    taking = set(re.findall(r'value\("(--[\w-]+)"\)', Path(__file__).read_text()))
    for before, word in zip(["", *args], args, strict=False):
        if word.startswith("--") and word not in known and before not in taking:
            return word
    return None


def broken_config() -> str | None:
    """Why the fake's configuration file does not parse, if it does not."""
    path = Path.home() / ".fake.json"
    try:
        json.loads(path.read_text())
    except FileNotFoundError:
        return None
    except ValueError as error:
        return f"fake: cannot parse {path}: {error}"
    return None


if __name__ == "__main__":
    if (found := unknown_option()) and not flag("--lax-options"):
        print(f"fake: unknown option {found}", file=sys.stderr)
        sys.exit(2)
    if flag("--git-status"):  # As an agent that puts the repository's state in its prompt.
        subprocess.run(["git", "status"], capture_output=True)
    if problem := broken_config():
        print(problem, file=sys.stderr)
        sys.exit(0 if flag("--config-exit-zero") else 2)
    if flag("--read-stdin") and not flag("--tui"):
        sys.stdin.read()  # As an agent that adds piped input to its prompt.
    sys.exit(interactive() if flag("--tui") else headless())
