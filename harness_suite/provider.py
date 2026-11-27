"""The fake provider: one loopback HTTP server for all three wire APIs.

It parses each request, checks it against the real API's rules, asks the
running check for a reply, and streams that reply with the requested timing
and faults. Every exchange is written to the trace before it is judged.
"""

import gzip
import json
import re
import socket
import struct
import threading
import time
import zlib
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlsplit

from .apis import anthropic, chat, responses
from .apis.common import Frame, Rejection, lone_surrogate
from .messages import BEFORE_FINISH, Error, Reply, Request
from .recipe import MODELS

APIS = {api.PATH: api for api in (chat, responses, anthropic)}
MAX_BODY = 32 * 1024 * 1024  # The largest request body Anthropic's API takes.
LONE_SURROGATE = "holds a lone UTF-16 surrogate"  # In the problem of a body that is not valid Unicode.


@dataclass
class Exchange:
    """One request and what the provider did with it."""

    number: int
    api: str
    started: float
    request: Request | None = None
    kind: str = "other"  # task, aux, sub (a sub-agent's), rejected, over-window, quit (sent by the suite's quit), other
    reply: Reply | None = None
    error: Error | None = None
    status: int = 200
    rejections: list[Rejection] = field(default_factory=list)  # The real API's rules the request breaks.
    finished: float | None = None
    outcome: str = "sent"  # sent, client-closed, cut
    launch: int = 0
    notes: dict = field(default_factory=dict)

    @property
    def problem(self) -> str | None:
        """The error the real API gives for the request: its first broken rule."""
        return self.rejections[0].text if self.rejections else None

    def summary(self) -> dict:
        reply = self.reply
        return {
            "number": self.number,
            "launch": self.launch,
            "api": self.api,
            "kind": self.kind,
            "started": round(self.started, 3),
            "finished": round(self.finished, 3) if self.finished else None,
            "status": self.status,
            "outcome": self.outcome,
            "problem": self.problem,
            "rules": [r.rule for r in self.rejections],
            "model": self.request.model if self.request else None,
            "tokens": self.request.tokens if self.request else None,
            "reply": None
            if reply is None
            else {
                "text": reply.text[:200],
                "thinking": bool(reply.thinking),
                "calls": [[c.id, c.name, c.arguments[:200]] for c in reply.calls],
                "stop": reply.stop_reason,
            },
            "error": self.error.kind if self.error else None,
            **self.notes,
        }


class Provider:
    """Serve `responder(exchange) -> Reply | Error` until closed. Each
    connection closes after its reply."""

    def __init__(self, directory: Path, responder, window: int, max_requests: int):
        self.directory = directory
        directory.mkdir(parents=True, exist_ok=True)
        self.responder = responder
        self.window = window
        self.max_requests = max_requests
        self.exchanges: list[Exchange] = []
        self.store: dict = {}
        self.launch = 0  # The launch a request comes from, unless its base URL says.
        self.failure: str | None = None  # A bug in the suite, not the harness.
        self.lock = threading.Lock()
        self.epoch, self.epoch_wall = time.monotonic(), time.time()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.provider = self
        # Closing waits for every handler, so the evidence is final when judged.
        self.server.daemon_threads = False
        self.server.block_on_close = True
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def now(self) -> float:
        return time.monotonic() - self.epoch

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()
        self.write_trace()

    def write_trace(self) -> None:
        with (self.directory / "trace.jsonl").open("w") as log:
            for exchange in self.exchanges:
                log.write(json.dumps(exchange.summary()) + "\n")

    def handle(
        self, path: str, headers: dict, raw: bytes, launch: int | None, prefixed: bool
    ) -> tuple[Exchange, object]:
        api = APIS.get(path)
        with self.lock:
            exchange = Exchange(len(self.exchanges) + 1, api.NAME if api else path, self.now(), launch=self.launch)
            exchange.notes["prefixed"] = prefixed
            if launch is not None:
                exchange.launch = launch
            self.exchanges.append(exchange)
            name = f"{exchange.number:03d}"
            (self.directory / f"{name}.request.json").write_bytes(raw)
            (self.directory / f"{name}.headers.json").write_text(json.dumps(masked(headers), indent=1))
            if api is None:
                exchange.status = 404
                return exchange, None
            if len(raw) > MAX_BODY:
                too_big = f"The body is {len(raw) / 1048576:.0f} MB, over the API's limit of 32 MB."
                exchange.rejections = [Rejection("body", too_big)]
                exchange.error = Error("too-large", message=exchange.problem)
                exchange.kind = "rejected"
                return exchange, api
            if exchange.number > self.max_requests:
                exchange.notes["limit"] = True
                exchange.error = Error("rate", retry_after="3600")
                exchange.status = 429
                return exchange, api
            try:
                body = json.loads(raw)
                exchange.request = api.parse(body, headers, self.store)
                exchange.rejections = api.problems(exchange.request, self.store)
                if found := lone_surrogate(body):
                    unicode = f"The body is not valid Unicode: {found} {LONE_SURROGATE}."
                    exchange.rejections = [Rejection("text", unicode), *exchange.rejections]
            except (ValueError, KeyError, TypeError, AttributeError) as error:
                malformed = f"The body is not a valid request ({type(error).__name__}: {error})"
                exchange.rejections = [Rejection("body", malformed)]
            if exchange.problem:
                exchange.error = Error("invalid", message=exchange.problem)
                exchange.kind = "rejected"
            else:
                outcome = self.responder(exchange)
                if isinstance(outcome, Error):
                    exchange.error = outcome
                else:
                    exchange.reply = outcome
            return exchange, api


# What nginx sends when its upstream fails.
GATEWAY_PAGE = b"<html><head><title>502 Bad Gateway</title></head><body><center><h1>502 Bad Gateway</h1></center><hr><center>nginx</center></body></html>"


class Handler(BaseHTTPRequestHandler):
    """One connection to the provider."""

    protocol_version = "HTTP/1.1"

    @property
    def provider(self) -> "Provider":
        return self.server.provider

    def log_message(self, *_):
        pass

    def connection_header(self, status: int) -> None:
        """Close the connection after the reply."""
        self.send_header("Connection", "close")
        self.close_connection = True

    def do_GET(self):
        path, _ = split(self.path)
        if path.rstrip("/").endswith("/models"):
            data = {
                "object": "list",
                "data": [{"id": m, "object": "model", "owned_by": "suite"} for m in MODELS],
            }
            self.reply_json(200, {}, data)
        else:
            self.reply_json(404, {}, {"error": {"message": "not found"}})

    def do_POST(self):
        path, launch = split(self.path)
        prefixed = launch is not None
        if not path.startswith("/v1/"):
            path = "/v1" + path
        try:
            raw = self.read_body()
            if path == "/v1/messages/count_tokens":
                self.reply_json(200, {}, {"input_tokens": len(raw) // 4 + 1})
                return
            headers = {k.lower(): v for k, v in self.headers.items()}
            exchange, api = self.provider.handle(path, headers, raw, launch, prefixed)
        except Exception as error:  # A suite bug must not look like a harness hang.
            self.provider.failure = f"{type(error).__name__}: {error}"
            self.reply_json(500, {}, {"error": {"message": "suite failure"}})
            return
        try:
            self.deliver(exchange, api)
        except (BrokenPipeError, ConnectionResetError):
            exchange.outcome = "client-closed"
        except Exception as error:
            self.provider.failure = f"{type(error).__name__}: {error}"
        finally:
            exchange.finished = self.provider.now()
            if exchange.outcome != "sent":
                self.close_connection = True

    def read_body(self) -> bytes:
        if self.headers.get("Transfer-Encoding", "").lower() == "chunked":
            raw = b""
            while True:
                size = int(self.rfile.readline().strip().split(b";")[0], 16)
                if not size:
                    self.rfile.readline()
                    break
                raw += self.rfile.read(size)
                self.rfile.readline()
        else:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
        if self.headers.get("Content-Encoding") == "gzip":
            raw = gzip.decompress(raw)
        return raw

    def reply_json(self, status: int, headers: dict, value) -> None:
        self.reply(status, json.dumps(value).encode(), "application/json", headers)

    def reply(self, status: int, data: bytes, kind: str, headers: dict) -> None:
        self.send_response(status)
        self.send_header("Content-Type", kind)
        for key, item in headers.items():
            self.send_header(key, item)
        self.send_header("Content-Length", str(len(data)))
        self.connection_header(status)
        self.end_headers()
        self.wfile.write(data)
        self.wfile.flush()

    def deliver(self, exchange: Exchange, api) -> None:
        name = f"{exchange.number:03d}"
        if api is None:
            self.reply_json(404, {}, {"error": {"message": "endpoint not implemented"}})
            return
        request = exchange.request
        if exchange.error and exchange.error.kind == "reset":
            # Linger 0: close with a TCP reset, as a dying proxy does.
            self.connection.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0))
            self.connection.close()
            exchange.outcome = "reset"
            return
        if exchange.error and exchange.error.kind == "gateway":
            exchange.status = 502
            self.reply(502, GATEWAY_PAGE, "text/html", {})
            return
        if exchange.error and (request is None or exchange.reply is None):
            error = exchange.error
            status, headers, value = api.error(error, request or dummy(api), self.provider.window)
            exchange.status = status
            folder = self.provider.directory
            (folder / f"{name}.response.json").write_text(json.dumps(value))
            (folder / f"{name}.response-headers.json").write_text(json.dumps({"status": status, **headers}, indent=1))
            self.reply_json(status, headers, value)
            return
        reply = exchange.reply
        stream = reply.stream
        if not request.stream:
            value = api.body(request, reply, exchange.number, self.provider.store)
            (self.provider.directory / f"{name}.response.json").write_text(json.dumps(value))
            self.reply_json(200, {}, value)
            return
        sent = shape(
            api.events(request, reply, exchange.number, self.provider.store), api, exchange, self.provider.window
        )
        # A gzip stream only to a client that accepts one, as a CDN does.
        packer = None
        if stream.framing == "gzip" and "gzip" in self.headers.get("Accept-Encoding", ""):
            packer = zlib.compressobj(wbits=31)
            exchange.notes["gzip"] = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        if packer:
            self.send_header("Content-Encoding", "gzip")
        # Chunked, as the real APIs stream: a connection that drops
        # mid-stream then breaks the HTTP body, not only the events.
        self.send_header("Transfer-Encoding", "chunked")
        self.connection_header(200)
        self.end_headers()
        with (self.provider.directory / f"{name}.response.sse").open("wb") as log:
            for item in sent:
                time.sleep(stream.pace)
                data = render(item, stream.framing)
                log.write(data)
                if packer:
                    data = packer.compress(data) + packer.flush(zlib.Z_SYNC_FLUSH)
                self.write(data, stream.chunk)
            if packer:
                self.write(packer.flush(), None)
        if stream.cut_after is not None:
            exchange.outcome = "cut"
            self.connection.shutdown(socket.SHUT_RDWR)
        else:
            self.wfile.write(b"0\r\n\r\n")  # The end of the chunked body.
            self.wfile.flush()

    def write(self, data: bytes, chunk: int | None) -> None:
        """Send stream data as HTTP chunks of at most `chunk` bytes."""
        if not data:
            return  # An empty chunk would end the body.
        size = chunk or len(data)
        for start in range(0, len(data), size):
            piece = data[start : start + size]
            self.wfile.write(b"%x\r\n%s\r\n" % (len(piece), piece))
            self.wfile.flush()
            if chunk:
                time.sleep(0.002)


def split(url: str) -> tuple[str, int | None]:
    """The API path, and the launch from a `/launch/N` prefix, as a base URL
    with a path has it."""
    path = urlsplit(url).path
    if found := re.fullmatch(r"/launch/(\d+)(/.*)", path):
        return found.group(2), int(found.group(1))
    return path, None


def render(item: Frame, framing: str) -> bytes:
    if item.data is None:
        text = f": {item.comment}\n\n"
    else:
        space = "" if framing == "no-space" else " "
        text = f"data:{space}{item.data}\n"
        if item.event:
            text = f"event:{space}{item.event}\n" + text
        text += "\n"
        if framing == "comments":
            text = ": suite heartbeat\n\n" + text
    if framing == "crlf":
        text = text.replace("\n", "\r\n")
    return text.encode()


def shape(frames: list[Frame], api, exchange: Exchange, window: int) -> list[Frame]:
    """The events of a reply as its stream sends them: with events a newer API
    version adds or Azure's filter chunk, without its finish events, cut short, ended by an error, or
    with one event's JSON broken."""
    request, stream = exchange.request, exchange.reply.stream
    if stream.unknown:
        frames = [frames[0], *api.unknown(request, exchange.number), *frames[1:]]
    if stream.azure_filter:
        frames = [*api.azure_filter(), *frames]
    if stream.drop_terminal:
        frames = [f for f in frames if not f.terminal]
    limit = next((n for n in (stream.cut_after, stream.error_after) if n is not None), None)
    if limit == BEFORE_FINISH:
        limit = next(i for i, f in enumerate(frames) if f.terminal)
    sent = frames if limit is None else frames[:limit]
    if stream.error_after is not None:
        sent = [*sent, api.stream_error(stream.error or Error("server"), request, window)]
    if stream.corrupt is not None and stream.corrupt < len(sent):
        broken = sent[stream.corrupt]
        sent[stream.corrupt] = Frame(broken.data[: len(broken.data) // 2], broken.event)
    return sent


SECRET_HEADERS = ("authorization", "x-api-key", "api-key", "proxy-authorization")


def masked(headers: dict) -> dict:
    """The headers as evidence, with credentials cut to their first characters."""
    return {k: f"{v[:12]}…" if k.lower() in SECRET_HEADERS and len(v) > 12 else v for k, v in headers.items()}


def dummy(api) -> Request:
    return Request(api.NAME, {}, {}, None, False, "", [], [], {})
