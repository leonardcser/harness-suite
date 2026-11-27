"""Freeze the official installers and build one versioned image per harness.

A recipe's `[install]` names its official installer and the command that runs
it, as one shell command or a list of steps. `installers/` holds each
installer, frozen at the recipe's version, so a later build installs as this
one did. The build serves them on the host loopback, at `${installers}`,
without changes. Extra `files` are frozen the same way, for an installer that
fetches release metadata of its own.
"""

import hashlib
import json
import re
import subprocess
import tempfile
import threading
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .recipe import ROOT, Recipe, render

INSTALLERS = ROOT / "installers"
BASE = "harness-suite-base"
BASE_DOCKERFILE = """\
FROM docker.io/library/ubuntu:26.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \\
    curl ca-certificates bash git build-essential procps ripgrep xz-utils python3 tmux \\
    && rm -rf /var/lib/apt/lists/*
"""
AGENT_DOCKERFILE = """\
FROM {base}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
RUN {install}
RUN useradd -m -o -u 1000 bench && mkdir -p /workspace && chown bench:bench /workspace && chmod 755 /root
RUN for f in /root/.local/bin/* /root/.opencode/bin/* /root/.pi/agent/bin/*; do [ ! -e "$f" ] || ln -sf "$f" /usr/local/bin/$(basename "$f"); done
ENV HOME=/home/bench USER=bench LOGNAME=bench TERM=xterm-256color COLORTERM=truecolor
WORKDIR /workspace
USER bench
"""


def fetch(url: str) -> bytes:
    return subprocess.check_output(
        ["curl", "--fail", "--silent", "--show-error", "--location", "--retry", "3", "--max-time", "120", url]
    )


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class Installers:
    """Serve the frozen installers of `recipes` on the host loopback. An
    installer that is not frozen at its recipe's version is downloaded first,
    in place of the one of an earlier version; commit `installers/` after."""

    def __init__(self, recipes: list[Recipe]):
        manifest = INSTALLERS / "manifest.json"
        self.manifest = json.loads(manifest.read_text()) if manifest.exists() else {}
        for r in recipes:
            if self.manifest.get(r.name, {}).get("version") != r.version:
                self.freeze(r)
                INSTALLERS.mkdir(exist_ok=True)
                manifest.write_text(json.dumps(self.manifest, indent=2, sort_keys=True) + "\n")
            for name, record in self.manifest[r.name]["files"].items():
                if digest((INSTALLERS / name).read_bytes()) != record["sha256"]:
                    raise ValueError(f"installers/{name} differs from its frozen sha256")

        class Handler(SimpleHTTPRequestHandler):
            def log_message(self, *_):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Handler, directory=str(INSTALLERS)))
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def freeze(self, r: Recipe) -> None:
        for name in self.manifest.get(r.name, {}).get("files", {}):
            (INSTALLERS / name).unlink(missing_ok=True)
        spec = r.install
        files = {f"{r.name}.sh": spec["installer"], **render(spec.get("files", {}), {"version": r.version})}
        records = {}
        for name, url in files.items():
            data = fetch(url)
            path = INSTALLERS / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records[name] = {"url": url, "sha256": digest(data), "bytes": len(data)}
        self.manifest[r.name] = {"version": r.version, "files": records}
        print(f"Froze the installer of {r.name} {r.version}; commit installers/", flush=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server.server_port}"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def build(tag: str, dockerfile: str) -> None:
    """Build `dockerfile` without the layer cache, so an image holds what the
    installers serve now and not what an earlier build downloaded."""
    with tempfile.TemporaryDirectory() as context:
        Path(context, "Dockerfile").write_text(dockerfile)
        subprocess.run(
            ["podman", "build", "--format=docker", "--network=host", "--no-cache", "-q", "-t", tag, context],
            check=True,
            stdout=subprocess.DEVNULL,
        )


def prepare(recipes: list[Recipe]) -> None:
    """Build the image of each recipe, and check that it runs the recipe's
    version."""
    installers = Installers(recipes)
    try:
        build(BASE, BASE_DOCKERFILE)
        for r in recipes:
            command = r.install["command"]
            steps = command if isinstance(command, list) else [command]
            install = " && ".join(render(steps, {"installers": installers.url, "version": r.version}))
            build(r.image, AGENT_DOCKERFILE.format(base=BASE, install=install))
            reported = subprocess.check_output(
                ["podman", "run", "--rm", r.image, r.data["command"][0], "--version"], text=True
            ).strip()
            if not re.search(rf"(?<![\d.]){re.escape(r.version)}(?![\d.])", reported):
                raise RuntimeError(f"Expected {r.name} {r.version}, got {reported}")
            print(f"Built {r.image}", flush=True)
    finally:
        installers.close()
