"""A terminal for an agent's TUI: a tmux pane of fixed size.

tmux answers the agent's terminal queries and keeps its screen. The suite
pastes prompts, presses keys, and reads the screen as a user sees it.
"""

import shlex
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from . import live

COLUMNS, ROWS = 120, 40
HISTORY = 2000  # Lines of scrollback that `screen` reads.
PASTE_MARK = 24  # Characters of a prompt's first line that show it arrived.


def mark(text: str) -> str:
    """The start of a prompt, which shows on the screen once the agent has it."""
    lines = text.strip().splitlines()
    return lines[0][:PASTE_MARK] if lines else ""


class Terminal:
    """The agent in its pane. `send(terminal, signal)` signals the agent, and
    `release` runs once the terminal ended."""

    def __init__(
        self,
        argv: list[str],
        log: Path,
        send: Callable[["Terminal", int], None],
        cwd: str | None = None,
        release: Callable[[], None] = lambda: None,
    ):
        self.tmux = ["tmux", "-L", f"suite-{live.token()}", "-f", "/dev/null"]
        self.log, self.send, self.release = log, send, release
        try:
            # One tmux call: the options hold before the agent can exit.
            self.run(
                "new-session", "-d", "-s", "suite", "-x", str(COLUMNS), "-y", str(ROWS),
                *(["-c", cwd] if cwd else []), shlex.join(argv), ";",
                "set", "-g", "remain-on-exit", "on", ";",
                "set", "-g", "history-limit", str(HISTORY), ";",
                "set", "-g", "status", "off", ";",
                "pipe-pane", "-O", "-t", "suite", f"cat >> {shlex.quote(str(log))}",
            )  # fmt: skip
        except BaseException:
            self.stop()
            raise
        live.add(self)

    def run(self, *args: str, data: str | None = None) -> str:
        return subprocess.run([*self.tmux, *args], input=data, check=True, capture_output=True, text=True).stdout

    def query(self, form: str) -> str:
        return self.run("display", "-p", "-t", "suite", form).strip()

    def poll(self) -> int | None:
        """The agent's exit status, or None while it runs."""
        try:
            if self.query("#{pane_dead}") != "1":
                return None
            return int(self.query("#{pane_dead_status}") or 0)
        except subprocess.CalledProcessError:
            return -1

    def screen(self) -> str:
        """The screen with its scrollback, as plain text: lines without trailing
        space, and no blank lines."""
        try:
            text = self.run("capture-pane", "-p", "-J", "-t", "suite", "-S", f"-{HISTORY}")
        except subprocess.CalledProcessError:
            return ""
        return "\n".join(line.rstrip() for line in text.splitlines() if line.strip())

    def written(self) -> int:
        return self.log.stat().st_size if self.log.exists() else 0

    def settle(self, quiet: float = 1.5, timeout: float = 60.0) -> bool:
        """Wait until the agent has drawn something and stopped drawing."""
        deadline = time.monotonic() + timeout
        size, changed = self.written(), time.monotonic()
        while time.monotonic() < deadline and self.poll() is None:
            time.sleep(0.1)
            now = self.written()
            if now != size:
                size, changed = now, time.monotonic()
            elif size and time.monotonic() - changed >= quiet:
                return True
        return False

    def shows(self, text: str, timeout: float = 60.0) -> bool:
        """Wait until the screen shows `text`."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline and self.poll() is None:
            if text in self.screen():
                return True
            time.sleep(0.2)
        return False

    def paste(self, text: str) -> None:
        """Paste `text` as one bracketed paste."""
        self.run("load-buffer", "-b", "prompt", "-", data=text)
        self.run("paste-buffer", "-p", "-d", "-b", "prompt", "-t", "suite")

    def type(self, text: str) -> None:
        self.run("send-keys", "-t", "suite", "-l", text)

    def keys(self, *names: str) -> None:
        self.run("send-keys", "-t", "suite", *names)

    def stop(self) -> int | None:
        """End the agent and the terminal. The exit status, if it had exited."""
        status = self.poll()
        if status is None:
            self.send(self, signal.SIGKILL)
        subprocess.run([*self.tmux, "kill-server"], capture_output=True)
        self.release()
        live.remove(self)
        return status
