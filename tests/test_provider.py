"""The fake provider's connections behave as a real server's do."""

import json
import socket
import tempfile
import unittest
from pathlib import Path

from harness_suite.messages import Reply
from harness_suite.provider import Provider
from harness_suite.recipe import MODELS

BODY = json.dumps({"model": MODELS[0], "messages": [{"role": "user", "content": "hi"}]}).encode()
REQUEST = (
    b"POST /v1/chat/completions HTTP/1.1\r\nHost: suite\r\nContent-Type: application/json\r\n"
    b"Content-Length: %d\r\n\r\n%s" % (len(BODY), BODY)
)


def answer(connection: socket.socket) -> bytes:
    """One whole response, by its Content-Length; b"" if the server closed."""
    data = b""
    while b"\r\n\r\n" not in data:
        if not (piece := connection.recv(65536)):
            return data
        data += piece
    head, _, body = data.partition(b"\r\n\r\n")
    size = int(next(line.split(b":")[1] for line in head.split(b"\r\n") if line.lower().startswith(b"content-length")))
    while len(body) < size:
        body += connection.recv(65536)
    return head


class Connections(unittest.TestCase):
    def test_each_connection_closes_after_its_reply_by_default(self):
        with tempfile.TemporaryDirectory() as folder:
            provider = Provider(Path(folder), lambda exchange: Reply(text="hi"), 200_000, 10)
            with provider, socket.create_connection(("127.0.0.1", provider.server.server_port)) as connection:
                connection.sendall(REQUEST)
                self.assertIn(b"Connection: close", answer(connection))
                self.assertEqual(connection.recv(1), b"")


if __name__ == "__main__":
    unittest.main()
