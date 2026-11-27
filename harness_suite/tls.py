"""HTTPS in front of the provider, as a company's TLS gateway serves it.

The front takes TLS on its own port and passes the bytes on to the plain
provider. Its certificate is signed by the suite's own CA, which no system
trusts: the agent trusts it only through the CA file that the standard
environment variables name, as a company that inspects TLS hands it out.
`failed` lists each handshake the agent broke off, with the reason.
"""

import select
import socket
import socketserver
import ssl
import subprocess
import tempfile
import threading
from functools import cache
from pathlib import Path

# The variables with which each runtime finds an extra CA: OpenSSL and Go,
# Node and Bun, Python's requests, curl.
VARIABLES = ("SSL_CERT_FILE", "NODE_EXTRA_CA_CERTS", "REQUESTS_CA_BUNDLE", "CURL_CA_BUNDLE")


@cache
def certificates() -> Path:
    """A folder with the CA's certificate, `ca.pem`, and the server's key and
    certificate for 127.0.0.1 and localhost, made once per suite run."""
    folder = Path(tempfile.mkdtemp(prefix="harness-suite-tls-"))

    def openssl(*words: str) -> None:
        subprocess.run(["openssl", *words], cwd=folder, check=True, capture_output=True)

    # The extensions that strict clients require of a CA and of a server.
    openssl("req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "30", "-subj", "/CN=Harness Suite CA",
            "-addext", "basicConstraints=critical,CA:TRUE", "-addext", "keyUsage=critical,keyCertSign,cRLSign",
            "-keyout", "ca.key", "-out", "ca.pem")  # fmt: skip
    openssl("req", "-newkey", "rsa:2048", "-nodes", "-subj", "/CN=127.0.0.1", "-keyout", "server.key",
            "-out", "server.csr")  # fmt: skip
    (folder / "san.ext").write_text(
        "subjectAltName=IP:127.0.0.1,DNS:localhost\n"
        "basicConstraints=critical,CA:FALSE\n"
        "keyUsage=critical,digitalSignature,keyEncipherment\n"
        "extendedKeyUsage=serverAuth\n"
        "authorityKeyIdentifier=keyid\n"
    )
    openssl("x509", "-req", "-in", "server.csr", "-CA", "ca.pem", "-CAkey", "ca.key", "-CAcreateserial",
            "-days", "30", "-extfile", "san.ext", "-out", "server.pem")  # fmt: skip
    return folder


class Tls:
    def __init__(self, provider_port: int, trusted: bool):
        """`trusted`: the agent gets the CA file, as when its company hands it out."""
        self.provider_port, self.trusted = provider_port, trusted
        self.failed: list[str] = []  # Each handshake the client broke off, and why.
        folder = certificates()
        self.ca = (folder / "ca.pem").read_text()
        context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        context.load_cert_chain(folder / "server.pem", folder / "server.key")
        outer = self

        class Handler(socketserver.BaseRequestHandler):
            def handle(self):
                try:
                    client = context.wrap_socket(self.request, server_side=True)
                except (ssl.SSLError, OSError) as error:
                    outer.failed.append(getattr(error, "reason", None) or type(error).__name__)
                    return
                with client, socket.create_connection(("127.0.0.1", outer.provider_port), timeout=10) as upstream:
                    relay(client, upstream)

        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True

    @property
    def provider_url(self) -> str:
        return f"https://127.0.0.1:{self.server.server_address[1]}"

    def environment(self, sandbox) -> dict[str, str]:
        """What tells an agent to trust the CA, in each spelling, with the CA
        file in its HOME; nothing if it does not get the CA."""
        if not self.trusted:
            return {}
        path = f"{sandbox.home}/suite-ca.pem"
        sandbox.write(path, self.ca, False)
        return dict.fromkeys(VARIABLES, path)

    def __enter__(self):
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def __exit__(self, *_):
        self.server.shutdown()
        self.server.server_close()


def relay(client: ssl.SSLSocket, upstream: socket.socket) -> None:
    """Pass bytes both ways until either side closes. Decrypted bytes the TLS
    socket already holds count as ready, which `select` cannot see."""
    upstream.settimeout(None)
    while True:
        ready = [client] if client.pending() else select.select([client, upstream], [], [], 120)[0]
        if not ready:
            return
        for source in ready:
            try:
                data = source.recv(65536)
            except (ssl.SSLError, OSError):
                return
            if not data:
                return
            (upstream if source is client else client).sendall(data)
