"""An HTTP proxy for agents, as a company network has.

The agent gets a provider URL whose host, `HOST`, nothing routes: only the
proxy reaches it, by passing the request on to the provider. It refuses every
other host, as a strict company proxy does, and keeps the agent's runs off
the internet; `seen` still lists each one. It takes plain requests in
absolute form (`POST http://host/path`) and `CONNECT` tunnels, one request
per connection.
"""

import select
import socket
import socketserver
import threading
from urllib.parse import urlsplit

# An address of TEST-NET-1 (RFC 5737), which no network routes. A name would
# need a resolver: one with a wildcard search domain answers every name.
HOST = "192.0.2.1"
VARIABLES = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy")


class Proxy:
    def __init__(self, provider_port: int):
        self.provider_port = provider_port
        self.seen: list[str] = []  # The target of each request, as "host:port".
        self.lock = threading.Lock()
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                outer.serve(self.request)

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    @property
    def provider_url(self) -> str:
        """The provider's address, which only the proxy reaches."""
        return f"http://{HOST}:{self.provider_port}"

    def environment(self, sandbox) -> dict[str, str]:
        """What tells an agent to use the proxy, in each spelling clients read."""
        return {**dict.fromkeys(VARIABLES, self.url), "NO_PROXY": "", "no_proxy": ""}

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()

    def serve(self, client: socket.socket) -> None:
        head = b""
        while b"\r\n\r\n" not in head:
            data = client.recv(65536)
            if not data:
                return
            head += data
        head, _, rest = head.partition(b"\r\n\r\n")
        lines = head.decode("latin-1").split("\r\n")
        method, target, version = lines[0].split(" ", 2)
        if method == "CONNECT":
            host, _, port = target.rpartition(":")
            upstream = self.connect(host, int(port), client)
            if upstream is None:
                return
            client.sendall(b"HTTP/1.1 200 Connection established\r\n\r\n")
            relay(client, upstream, rest)
            return
        parts = urlsplit(target)
        if not parts.hostname:
            client.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return
        upstream = self.connect(parts.hostname, parts.port or 80, client)
        if upstream is None:
            return
        path = parts.path + (f"?{parts.query}" if parts.query else "")
        kept = [line for line in lines[1:] if line.split(":", 1)[0].lower() not in ("connection", "proxy-connection")]
        request = "\r\n".join([f"{method} {path or '/'} {version}", *kept, "Connection: close", "", ""])
        relay(client, upstream, request.encode("latin-1") + rest)

    def connect(self, host: str, port: int, client: socket.socket) -> socket.socket | None:
        with self.lock:
            self.seen.append(f"{host}:{port}")
        if host != HOST:
            client.sendall(b"HTTP/1.1 403 Forbidden\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return None
        try:
            return socket.create_connection(("127.0.0.1", self.provider_port), timeout=10)
        except OSError:
            client.sendall(b"HTTP/1.1 502 Bad Gateway\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
            return None


def relay(client: socket.socket, upstream: socket.socket, first: bytes) -> None:
    """Pass bytes both ways until either side closes."""
    with upstream:
        upstream.settimeout(None)
        if first:
            upstream.sendall(first)
        sockets = [client, upstream]
        while True:
            ready, _, _ = select.select(sockets, [], [], 120)
            if not ready:
                return
            for source in ready:
                try:
                    data = source.recv(65536)
                except OSError:  # Either side reset the connection.
                    return
                if not data:
                    return
                (upstream if source is client else client).sendall(data)
