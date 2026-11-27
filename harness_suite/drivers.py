"""How the suite drives an agent through a scenario's launches.

Headless, each launch is one process that runs one prompt and exits. In the
TUI, a launch that continues the session is the next prompt pasted into the
same TUI, and a turn ends when the agent goes quiet. Both watch a launch with
the same loop: the agent ends it, or the suite does at its deadline, at the
request limit, once the outcome is known, or by killing it as a crash would.
Launches run one after the other.
"""

import json
import os
import signal
import socket
import subprocess
import time
from dataclasses import replace
from pathlib import Path

from .evidence import KEPT_OUTPUT, LaunchRecord, Run
from .recipe import MAX_OUTPUT, MODELS, merge_json
from .script import Launch
from .terminal import Terminal, mark

POLL = 0.05
TOOL_QUIET = 30.0  # TUI: a running tool may keep the agent quiet this long.
PROMPT_TRIES = 4  # TUI: times the suite sends a prompt the agent lost.
PASTE_QUIET = 1.0  # TUI: seconds without drawing that show the agent has taken a paste.
PASTE_SHOWN = 15.0  # TUI: at most this long to wait for that.
PROMPT_TAKEN = 10.0  # TUI: seconds for a request that shows the agent took the prompt.
LOOK_EVERY = 1.0  # TUI: seconds between looks at the screen during a launch.
DONE_QUIET = 2.0  # TUI: after the script's last step, a turn ends sooner.


class Driver:
    def __init__(self, scenario, model, sandbox, provider, directory: Path, servers: list[dict], front=None):
        self.scenario, self.model, self.recipe = scenario, model, scenario.recipe
        self.servers = servers  # The MCP servers, as the recipe's templates see them.
        self.sandbox, self.provider, self.directory = sandbox, provider, directory
        # What alone reaches the provider, if the scenario has it: an HTTP
        # proxy or a TLS front. It gives the base URL and the environment.
        self.front = front
        self.run: Run = model.run
        self.events = EventLog(sandbox.host_path(sandbox.fixture_dir) / "calls.jsonl")

    def drive(self) -> None:
        try:
            for index, launch in enumerate(self.scenario.launches()):
                record = LaunchRecord(index)
                self.run.launches.append(record)
                self.start(index, launch, record)
                if record.termination == "request-limit" or not self.goes_on(record):
                    break
        finally:
            self.close()

    def start(self, index: int, launch: Launch, record: LaunchRecord) -> None:
        self.provider.launch = index
        record.started = self.provider.now()
        try:
            self.launch(index, launch, record)
        finally:
            record.ended = self.provider.now()
            (self.directory / f"launch-{index}.json").write_text(json.dumps(record.about(), indent=2))

    def watch(self, process, launch: Launch, record: LaunchRecord) -> None:
        deadline = time.monotonic() + (launch.timeout or self.scenario.timeout)
        while True:
            time.sleep(POLL)
            if process.poll() is not None:
                if record.killed is None:
                    record.termination = "exited"
                return
            if self.limited():
                record.termination, record.stopped = "request-limit", self.provider.now()
                return
            if time.monotonic() >= deadline:
                record.termination, record.stopped = "deadline", self.provider.now()
                return
            self.look()
            if self.scenario.settled(self.run):
                record.termination, record.stopped = "settled", self.provider.now()
                return
            self.observe(record)
            if ending := self.over(record):
                record.termination = ending
                return
            if launch.kill_when and record.killed is None and launch.kill_when(self.run):
                process.signal(signal.SIGKILL)
                record.termination, record.killed = "killed", self.provider.now()

    def look(self) -> None:
        """Bring the fixture's events in the evidence up to date."""
        self.run.fixture = self.events.read()

    def observe(self, record: LaunchRecord) -> None:
        """Look at the agent while the launch runs."""

    def limited(self) -> bool:
        return any(e.notes.get("limit") for e in self.provider.exchanges)

    def prepare(self, launch: Launch, index: int, record: LaunchRecord):
        """Write the launch's files; return what the sandbox starts."""
        sandbox, scenario = self.sandbox, self.scenario
        settings = dict(launch.settings)
        if launch.resume and scenario.frontend == "headless":
            settings["continue"] = True
        model = launch.model or MODELS[0]
        variables = {
            "home": sandbox.home,
            "workspace": sandbox.workspace,
            "prompt_file": f"{sandbox.home}/prompt-{index}.txt",
            "provider_url": self.provider_url(index),
            "model": model,
            "model_b": MODELS[1] if model != MODELS[1] else MODELS[0],
            "context_window": scenario.window,
            "max_output_tokens": MAX_OUTPUT,
            "python": sandbox.python,
            "fixture": sandbox.fixture,
            "fixture_dir": sandbox.fixture_dir,
            "free_port": free_port(),
        }
        api = record.api = launch.api or scenario.api
        # A launch with a broken file writes every file, as the first does, so
        # the file it breaks is the one the agent reads.
        first = index == 0 or launch.broken_file
        prepared = self.recipe.prepare(
            variables, api, scenario.source, settings, launch.prompt, first, scenario.frontend, self.servers
        )
        if self.front is not None:
            prepared = replace(prepared, environment={**prepared.environment, **self.front.environment(sandbox)})
        if launch.open_stdin and prepared.stdin is not None:
            # Stdin is the script's own pipe, so the prompt goes in the arguments.
            prepared = replace(prepared, argv=[*prepared.argv, prepared.stdin], stdin=None)
        if launch.options:
            prepared = replace(prepared, argv=[*prepared.argv, *launch.options])
        if launch.broken_file:
            prepared = replace(prepared, files=broken(prepared.files))
            record.broken = next((path for path, _, mode in prepared.files if mode == "replace"), None)
        self.model.models.add(prepared.model)
        sandbox.write(variables["prompt_file"], launch.prompt, False)
        for path, text, mode in prepared.files:
            if mode == "merge":
                text = merged(sandbox.host_path(path), text)
            sandbox.write(path, text, mode == "append")
        record.argv, record.environment = prepared.argv, prepared.environment
        return prepared, sandbox.host_path(variables["prompt_file"])

    def provider_url(self, index: int) -> str:
        base = self.front.provider_url if self.front else self.provider.url
        return base + (f"/launch/{index}" if self.scenario.prefixed else "")

    # What each frontend writes.

    def launch(self, index: int, launch: Launch, record: LaunchRecord) -> None:
        raise NotImplementedError

    def over(self, record: LaunchRecord) -> str | None:
        """How the launch ended though the process still runs, or None."""
        return None

    def goes_on(self, record: LaunchRecord) -> bool:
        return True

    def close(self) -> None:
        pass


class Headless(Driver):
    def launch(self, index, launch, record):
        prepared, prompt_file = self.prepare(launch, index, record)
        out = (self.directory / f"launch-{index}.stdout").open("wb")
        err = (self.directory / f"launch-{index}.stderr").open("wb")
        stdin = held = None
        if prepared.stdin is not None:
            stdin = prompt_file.open("rb")
        elif launch.open_stdin:
            reader, writer = os.pipe()
            stdin, held = os.fdopen(reader, "rb"), os.fdopen(writer, "wb")
        process = None
        try:
            # Without a prompt or a held pipe on stdin, the agent reads an empty stdin.
            process = self.sandbox.start(prepared.argv, prepared.environment, stdin or subprocess.DEVNULL, out, err)
            if held:
                stdin.close()  # The agent has the read end; the suite holds only the write end.
                stdin = None
            self.watch(process, launch, record)
        finally:
            if process is not None:
                record.exit = process.stop()
            for file in (out, err, stdin, held):
                if file:
                    file.close()
        with (self.directory / f"launch-{index}.stdout").open(errors="replace") as out:
            record.stdout = out.read(KEPT_OUTPUT + 1)
        record.stderr = (self.directory / f"launch-{index}.stderr").read_text(errors="replace")[-4000:]


class Tui(Driver):
    def __init__(self, *args):
        super().__init__(*args)
        self.terminal, self.log = None, None
        self.spec = self.recipe.tui
        self.looked, self.seen = 0.0, set()

    def launch(self, index, launch, record):
        terminal = self.terminal
        if terminal is None or not launch.resume or terminal.poll() is not None:
            if terminal is not None:
                self.quit()
            prepared, _ = self.prepare(launch, index, record)
            self.log = self.directory / f"terminal-{index}.log"
            terminal = self.terminal = self.sandbox.start_tui(prepared.argv, prepared.environment, self.log)
            terminal.settle(timeout=60)
            if ready := self.spec.get("ready"):
                terminal.shows(ready)
        self.seen = set()
        if terminal.poll() is None and self.prompt(terminal, launch, record):
            self.watch(terminal, launch, record)
        screen = terminal.screen()
        self.keep(record, screen)
        (self.directory / f"launch-{index}.screen.txt").write_text(screen)
        (self.directory / f"launch-{index}.shown.txt").write_text("\n".join(record.shown))
        record.stdout = screen[-4000:]
        record.written = self.log.read_bytes() if self.log.is_file() else b""

    def prompt(self, terminal: Terminal, launch: Launch, record: LaunchRecord) -> bool:
        """Paste the prompt and press Enter, and again if the agent lost it;
        whether it took the prompt. An agent that has drawn its screen may
        still be starting, and drop the paste or the Enter. Enter waits until
        the screen is quiet, not until the paste shows: an agent may show a
        long paste as a placeholder."""
        start = mark(launch.prompt)
        for _ in range(PROMPT_TRIES):
            before = terminal.screen().count(start)
            record.prompts += 1
            terminal.paste(launch.prompt)
            # Enter once the agent has drawn the paste, as a user presses it: a
            # key that comes sooner may become part of the paste, or pick from
            # a list that the paste opens.
            terminal.settle(quiet=PASTE_QUIET, timeout=PASTE_SHOWN)
            terminal.keys("Enter")
            if self.taken(terminal, record, start, before):
                return True
        if not self.scenario.prompt_may_be_lost:
            self.run.unmeasured = (
                f"The TUI lost prompt {record.index + 1} each of the {record.prompts} times the suite sent it."
            )
        return False

    def taken(self, terminal: Terminal, record: LaunchRecord, start: str, before: int) -> bool:
        """Whether the agent took the prompt: it sent a request within
        `PROMPT_TAKEN` seconds, or the prompt still shows on the screen."""
        deadline = time.monotonic() + PROMPT_TAKEN
        while time.monotonic() < deadline and terminal.poll() is None:
            if any(e.launch == record.index for e in self.provider.exchanges):
                return True
            self.observe(record)
            time.sleep(POLL)
        return terminal.poll() is not None or terminal.screen().count(start) > before

    def observe(self, record):
        """Keep each line the screen shows, so that a message the agent shows
        for a moment, such as why it retries, counts as said."""
        if time.monotonic() - self.looked >= LOOK_EVERY:
            self.looked = time.monotonic()
            self.keep(record, self.terminal.screen())

    def keep(self, record: LaunchRecord, screen: str) -> None:
        for line in screen.splitlines():
            if line not in self.seen:
                self.seen.add(line)
                record.shown.append(line)

    def over(self, record):
        """The turn is over when the agent has sent a request since the
        prompt, none is open, and no request has come and no fixture tool
        has run for `quiet` seconds."""
        mine = [e for e in self.provider.exchanges if e.launch == record.index]
        if not mine or any(e.finished is None for e in mine):
            return None
        self.look()
        ran = [self.run.clock(e["time"]) for e in self.run.fixture if e.get("event", "call") in ("call", "beat")]
        last = max([*(e.finished for e in mine), *ran])
        quiet = self.scenario.quiet
        track = self.model.track(record.index)
        if track.pending is not None:
            quiet = max(quiet, TOOL_QUIET)
        elif track.done:
            quiet = min(quiet, DONE_QUIET)
        if self.provider.now() - last < quiet:
            return None
        return "idle"

    def goes_on(self, record):
        """A next prompt can go to the TUI: it still runs."""
        return self.terminal is not None and self.terminal.poll() is None

    def quit(self) -> int | None:
        """Quit as a user does, then stop what is left. A request that carries
        the quit command to the model sent a prompt that Enter did not: it is
        the suite's, not the case's."""
        terminal, self.terminal = self.terminal, None
        command = self.spec.get("quit", "/quit")
        if terminal.poll() is None:
            sent = len(self.provider.exchanges)
            terminal.type(command)
            time.sleep(0.5)
            terminal.keys("Enter")
            deadline = time.monotonic() + 10
            while terminal.poll() is None and time.monotonic() < deadline:
                time.sleep(0.1)
            for exchange in self.provider.exchanges[sent:]:
                if exchange.request and any(command in text for text in exchange.request.user_texts):
                    exchange.kind = "quit"
        return terminal.stop()

    def close(self):
        if self.terminal is None:
            return
        status = self.quit()
        if self.run.launches and self.run.last.termination == "exited":
            self.run.last.exit = status


# Not TOML, JSON or YAML, and plain to spot in an error message.
BROKEN_LINE = "\n]]] SUITE_BROKEN_LINE [[[\n"


def broken(files: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """The files, with a line no format parses added to the first that the
    launch writes whole."""
    first = next((n for n, (_, _, mode) in enumerate(files) if mode == "replace"), None)
    return [(p, t + BROKEN_LINE if n == first else t, m) for n, (p, t, m) in enumerate(files)]


def free_port() -> int:
    """A free TCP port on the host, for an agent that serves on a fixed port:
    containers share the host's network, and cases run in parallel."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class EventLog:
    """The fixture's events, read as the fixture appends them: each read
    parses only the lines written since the last."""

    def __init__(self, path: Path):
        self.path, self.offset, self.events = path, 0, []

    def read(self) -> list[dict]:
        try:
            with self.path.open("rb") as file:
                file.seek(self.offset)
                data = file.read()
        except FileNotFoundError:
            return self.events
        whole = data.rfind(b"\n") + 1  # A line still being written waits for the next read.
        self.offset += whole
        for line in data[:whole].splitlines():
            try:
                self.events.append(json.loads(line))
            except ValueError:
                pass
        return self.events


def merged(path: Path, text: str) -> str:
    """A JSON file's text merged into the file already at `path`: a later
    launch adds to what an earlier one, or the agent, wrote."""
    if not path.is_file():
        return text
    try:
        data = json.loads(path.read_text())
    except ValueError:
        return text
    merge_json(data, json.loads(text))
    return json.dumps(data, indent=2)
