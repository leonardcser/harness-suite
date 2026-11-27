"""The fixture's MCP server speaks each HTTP transport as the protocol says."""

import contextlib
import json
import queue
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from harness_suite import fixture

INITIALIZE = {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "initialize",
    "params": {"protocolVersion": "2025-06-18", "capabilities": {}, "clientInfo": {"name": "test", "version": "1"}},
}
LIST = {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
CALL = {
    "jsonrpc": "2.0",
    "id": 3,
    "method": "tools/call",
    "params": {"name": "suite_fixture", "arguments": {"kind": "text", "token": "t1"}},
}
INITIALIZED = {"jsonrpc": "2.0", "method": "notifications/initialized"}


def serving(test: unittest.TestCase, transport: str, name: str = "alpha") -> str:
    """Start the fixture's server over `transport`; its base URL."""
    folder = Path(tempfile.mkdtemp())
    process = subprocess.Popen([sys.executable, fixture.__file__, transport, str(folder), name])
    test.addCleanup(process.wait)
    test.addCleanup(process.kill)
    port = folder / f"{name}.port"
    deadline = time.monotonic() + 10
    while not port.exists():
        test.assertLess(time.monotonic(), deadline, "the server did not start")
        time.sleep(0.02)
    return f"http://127.0.0.1:{port.read_text()}"


def post(url: str, value, session: str | None = None):
    """(status, headers, body) of a POST."""
    headers = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream"}
    if session:
        headers["Mcp-Session-Id"] = session
    try:
        with urlopen(Request(url, json.dumps(value).encode(), headers), timeout=10) as response:
            return response.status, response.headers, response.read().decode()
    except HTTPError as error:
        return error.code, error.headers, error.read().decode()


def data(stream: str) -> list[dict]:
    return [json.loads(line[5:]) for line in stream.splitlines() if line.startswith("data:")]


class Streamable(unittest.TestCase):
    def setUp(self):
        self.url = serving(self, "http") + "/mcp"

    def open(self) -> str:
        status, headers, body = post(self.url, INITIALIZE)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["id"], 1)
        session = headers["Mcp-Session-Id"]
        self.assertEqual(post(self.url, INITIALIZED, session)[0], 202)
        return session

    def test_a_session_lists_and_calls_the_named_server(self):
        session = self.open()
        status, _, body = post(self.url, LIST, session)
        self.assertEqual(status, 200)
        tool = json.loads(body)["result"]["tools"][0]
        self.assertIn("Server alpha.", tool["description"])
        status, headers, body = post(self.url, CALL, session)
        self.assertEqual(headers.get_content_type(), "text/event-stream")
        answer = data(body)[-1]
        self.assertEqual(answer["id"], 3)
        self.assertIn(fixture.marker("t1"), answer["result"]["content"][0]["text"])

    def test_requests_without_the_session_are_refused(self):
        self.assertEqual(post(self.url, LIST)[0], 400, "before initialize")
        session = self.open()
        self.assertEqual(post(self.url, LIST)[0], 400, "no session ID")
        self.assertEqual(post(self.url, LIST, "other")[0], 404, "an unknown session ID")
        request = Request(self.url, method="DELETE", headers={"Mcp-Session-Id": session})
        urlopen(request, timeout=10).close()
        self.assertEqual(post(self.url, LIST, session)[0], 404, "an ended session")
        self.assertEqual(post(self.url, LIST)[0], 400, "after the session ended")


class Sse(unittest.TestCase):
    def test_answers_come_on_the_stream(self):
        base = serving(self, "sse")
        response = urlopen(Request(base + "/sse", headers={"Accept": "text/event-stream"}), timeout=10)
        self.addCleanup(response.close)
        events: queue.Queue = queue.Queue()

        def read():
            kind = None
            with contextlib.suppress(OSError):  # The test closes the stream.
                for raw in response:
                    line = raw.decode().strip()
                    if line.startswith("event:"):
                        kind = line[6:].strip()
                    elif line.startswith("data:"):
                        events.put((kind, line[5:].strip()))

        threading.Thread(target=read, daemon=True).start()
        kind, endpoint = events.get(timeout=10)
        self.assertEqual(kind, "endpoint")
        self.assertEqual(post(base + "/messages?sessionId=other", INITIALIZE)[0], 404)
        for message in (INITIALIZE, INITIALIZED, LIST):
            self.assertEqual(post(base + endpoint, message)[0], 202)
        answers = [json.loads(events.get(timeout=10)[1]) for _ in range(2)]
        self.assertEqual([a["id"] for a in answers], [1, 2])
        self.assertIn("Server alpha.", answers[1]["result"]["tools"][0]["description"])


if __name__ == "__main__":
    unittest.main()
