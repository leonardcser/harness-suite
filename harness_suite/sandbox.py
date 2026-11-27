"""Where an agent runs: on the host, or in its pinned container image.

Both give the agent a fresh HOME and workspace per case, with the suite's
fixture script in the workspace, at `.suite/fixture.py`, where a command that
a user runs would be. Containers run the images that `images.py` builds
(`harness-suite-NAME:VERSION`), with the harness's official install.

A sandbox starts the agent headless as a `Process`, or in its TUI as a
`Terminal`. Both have `poll`, `signal` and `stop`, and stay in `live` until
stopped. Each agent runs in its own cgroup in the suite's slice, under its own
memory cap (`limits`).
"""

import os
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
from pathlib import Path

from . import limits, live
from .terminal import Terminal

SUITE = Path(__file__).resolve().parent


def base_environment(home: str, tty: bool) -> dict:
    return {
        "HOME": home,
        "XDG_CONFIG_HOME": f"{home}/.config",
        "XDG_DATA_HOME": f"{home}/.local/share",
        "XDG_CACHE_HOME": f"{home}/.cache",
        "XDG_STATE_HOME": f"{home}/.local/state",
        "TMPDIR": f"{home}/tmp",
        "LANG": "C.UTF-8",
        "TERM": "xterm-256color" if tty else "dumb",
        "NO_COLOR": "1",
        "SUITE_API_KEY": "suite-placeholder",
        "DISABLE_TELEMETRY": "1",
        "DISABLE_AUTOUPDATER": "1",
        "DISABLE_ERROR_REPORTING": "1",
    }


class Process:
    """A headless agent. `release` runs once it ended."""

    def __init__(self, popen: subprocess.Popen, send, release=lambda: None):
        self.popen, self.send, self.release = popen, send, release
        live.add(self)

    def poll(self) -> int | None:
        return self.popen.poll()

    def signal(self, sig: int) -> None:
        if self.popen.poll() is None:
            try:
                self.send(sig)
            except ProcessLookupError:
                pass

    def stop(self) -> int | None:
        """End the agent; its exit status."""
        self.signal(signal.SIGKILL)
        try:
            status = self.popen.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self.popen.kill()
            status = self.popen.wait()
        self.release()
        live.remove(self)
        return status


class Sandbox:
    """Fresh directories for one case. Paths are as the agent sees them."""

    home: str
    workspace: str
    python: str

    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix="harness-suite-"))
        self.host_home = self.root / "home"
        self.host_workspace = self.root / "workspace"
        for path in (self.host_home / "tmp", self.host_workspace / ".suite"):
            path.mkdir(parents=True)
        shutil.copy(SUITE / "fixture.py", self.host_workspace / ".suite" / "fixture.py")

    @property
    def fixture_dir(self) -> str:
        return f"{self.workspace}/.suite"

    @property
    def fixture(self) -> str:
        return f"{self.fixture_dir}/fixture.py"

    def host_path(self, seen: str) -> Path:
        """The host path of a path inside the sandbox."""
        for inner, outer in ((self.home, self.host_home), (self.workspace, self.host_workspace)):
            if seen == inner or seen.startswith(inner + "/"):
                return outer / seen[len(inner) :].lstrip("/")
        raise ValueError(f"Recipe path outside HOME and workspace: {seen}")

    def write(self, seen: str, text: str, append: bool) -> None:
        """Write a file of a launch. A whole file is replaced at once, so an
        agent never reads it half written."""
        path = self.host_path(seen)
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if append:
            with path.open("a") as file:
                file.write(text)
            path.chmod(0o600)
            return
        replace_atomically(path, text)

    def run(self, argv: list[str], timeout: float = 60.0) -> subprocess.CompletedProcess:
        """Run a short command as the agent's user would, in the workspace."""
        raise NotImplementedError

    def close(self) -> None:
        shutil.rmtree(self.root, ignore_errors=True)


class Host(Sandbox):
    def __init__(self, command: list[str]):
        super().__init__()
        self.home, self.workspace, self.python = str(self.host_home), str(self.host_workspace), sys.executable
        self.command = command

    def environment(self, environment: dict, tty: bool) -> dict:
        return {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), **base_environment(self.home, tty), **environment}

    def words(self, argv: list[str], environment: dict, tty: bool) -> tuple[str, list[str]]:
        """The agent's scope, and its command in that capped scope with only
        its environment. The scope comes first: systemd-run needs the suite's
        environment to reach the user manager."""
        unit = f"harness-suite-{live.token()}"
        command = [*self.command, *argv[1:]] if self.command else argv
        variables = [f"{k}={v}" for k, v in self.environment(environment, tty).items()]
        return unit, [*limits.scope(limits.AGENT_MEMORY, unit), "env", "-i", *variables, *command]

    def run(self, argv, timeout=60.0):
        environment = self.environment({}, False)
        return subprocess.run(argv, cwd=self.workspace, env=environment, capture_output=True, timeout=timeout)

    def start(self, argv: list[str], environment: dict, stdin, out, err) -> Process:
        unit, words = self.words(argv, environment, False)
        popen = subprocess.Popen(
            words,
            cwd=self.workspace,
            stdin=stdin,
            stdout=out,
            stderr=err,
            start_new_session=True,
        )
        return Process(popen, lambda sig: os.killpg(popen.pid, sig), lambda: limits.release(unit))

    def start_tui(self, argv: list[str], environment: dict, log: Path) -> Terminal:
        unit, words = self.words(argv, environment, True)

        def send(terminal: Terminal, sig: int) -> None:
            try:
                os.killpg(int(terminal.query("#{pane_pid}")), sig)
            except (ProcessLookupError, ValueError, subprocess.CalledProcessError):
                pass

        return Terminal(words, log, send, self.workspace, lambda: limits.release(unit))


class Container(Sandbox):
    def __init__(self, image: str):
        super().__init__()
        self.home, self.workspace, self.python = "/home/bench", "/workspace", "python3"
        self.image = image

    def words(self, argv: list[str], environment: dict, tty: bool) -> tuple[str, list[str]]:
        name = f"harness-suite-{live.token()}"
        words = [
            "podman",
            "run",
            "--rm",
            "-i",
            *(["-t"] if tty else []),
            "--name",
            name,
            "--cidfile",
            self.cidfile(name),
        ]
        words += ["--network=host", "--userns=keep-id", "--init", *limits.container(), "-w", self.workspace]
        words += ["-v", f"{self.host_home}:{self.home}", "-v", f"{self.host_workspace}:{self.workspace}"]
        for key, value in {**base_environment(self.home, tty), **environment}.items():
            words += ["-e", f"{key}={value}"]
        return name, [*words, self.image, *argv]

    def run(self, argv, timeout=60.0):
        _, words = self.words(argv, {}, False)
        return subprocess.run(words, stdin=subprocess.DEVNULL, capture_output=True, timeout=timeout)

    def start(self, argv: list[str], environment: dict, stdin, out, err) -> Process:
        name, words = self.words(argv, environment, False)
        popen = subprocess.Popen(words, stdin=stdin, stdout=out, stderr=err, start_new_session=True)
        return Process(popen, signaller(name))

    def start_tui(self, argv: list[str], environment: dict, log: Path) -> Terminal:
        name, words = self.words(argv, environment, True)
        send = signaller(name)
        return Terminal(words, log, lambda _, sig: send(sig))

    def cidfile(self, name: str) -> str:
        return str(self.root / f"{name}.cid")


def signaller(name: str):
    def send(sig: int) -> None:
        if sig == signal.SIGKILL:
            subprocess.run(["podman", "rm", "-f", "-t", "0", name], capture_output=True)
        else:
            subprocess.run(["podman", "kill", "--signal", signal.Signals(sig).name, name], capture_output=True)

    return send


def image_exists(image: str) -> bool:
    return subprocess.run(["podman", "image", "exists", image], capture_output=True).returncode == 0


def replace_atomically(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    temporary.write_text(text)
    temporary.chmod(0o600)
    os.replace(temporary, path)
