"""What a judge can look at for one case. Times are seconds on the provider's
clock, which starts when the provider does."""

from dataclasses import dataclass, field

from . import limits
from .messages import Call, Request, Result
from .provider import Exchange

KEPT_OUTPUT = 8 * 1024**2  # Characters of headless stdout a launch record keeps.


@dataclass
class Served:
    """A step as the provider sent it."""

    step: int
    exchange: int
    calls: list[Call]
    tokens: list[str]
    track: int = 0


@dataclass
class LaunchRecord:
    index: int
    started: float = 0.0
    # exited: the process ended. idle: the TUI's turn went quiet. deadline,
    # request-limit or settled (the outcome was known): the suite stopped it.
    # killed: the suite killed it, as `Launch.kill_when` asks.
    termination: str = "exited"
    exit: int | None = None
    killed: float | None = None  # When the suite killed it.
    broken: str | None = None  # The configuration file the suite broke.
    stopped: float | None = None  # When the suite stopped it: at its deadline, the request limit, or once settled.
    ended: float = 0.0  # When the suite stopped watching the launch.
    stdout: str = ""  # Headless: its output, from the start. TUI: the screen.
    stderr: str = ""  # Headless: the end of its errors.
    shown: list[str] = field(default_factory=list)  # TUI: each line on screen during the launch, first seen first.
    written: bytes = b""  # TUI: every byte the agent wrote to its terminal, up to the launch's end.
    prompts: int = 0  # TUI: how often the suite sent the prompt; again when the agent lost it.
    api: str = ""
    argv: list[str] | None = None  # What the suite started; None when the prompt went to the running TUI.
    environment: dict[str, str] = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return self.ended - self.started

    def about(self) -> dict:
        """The launch as `launch-N.json` saves it."""
        started = {"argv": self.argv, "environment": self.environment} if self.argv is not None else {"resumed": True}
        return {
            "api": self.api,
            **started,
            "termination": self.termination,
            "exit": self.exit,
            "started": round(self.started, 3),
            "killed": self.killed and round(self.killed, 3),
            "stopped": self.stopped and round(self.stopped, 3),
            "ended": round(self.ended, 3),
            **({"prompts": self.prompts} if self.prompts else {}),
            **({"broken": self.broken} if self.broken else {}),
        }


@dataclass
class Run:
    variant: str
    api: str
    source: str
    frontend: str
    epoch: float = 0.0  # Wall-clock time of the provider's 0.
    workspace: str = ""  # The workspace, as the agent sees it.
    exchanges: list[Exchange] = field(default_factory=list)
    served: list[Served] = field(default_factory=list)
    fixture: list[dict] = field(default_factory=list)  # Every event the fixture logged.
    launches: list[LaunchRecord] = field(default_factory=list)
    finished: set[int] = field(default_factory=set)  # The tracks whose last step was served.
    missing_at: float | None = None  # When the first request without the fixture tool came.
    unmeasured: str | None = None  # Why the run cannot show what its checks look for.
    proxied: list[str] = field(default_factory=list)  # The proxy's targets, as "host:port", if it ran.
    handshakes_failed: list[str] = field(default_factory=list)  # Why TLS handshakes failed, if a TLS front ran.
    outputs: dict[str, bytes | None] = field(default_factory=dict)  # Workspace files after the run.

    @property
    def finished_script(self) -> bool:
        """The main track's last step was served."""
        return 0 in self.finished

    @property
    def tasks(self) -> list[Exchange]:
        return [e for e in self.exchanges if e.kind == "task"]

    @property
    def requests(self) -> list[Request]:
        return [e.request for e in self.tasks]

    @property
    def reached(self) -> bool:
        """Whether the agent reached the provider: a request, or a TLS handshake."""
        return bool(self.handshakes_failed) or any(e.request for e in self.exchanges)

    @property
    def calls(self) -> list[Call]:
        return [call for served in self.served for call in served.calls]

    @property
    def fixture_calls(self) -> list[dict]:
        return [e for e in self.fixture if e.get("event", "call") == "call"]

    def rpc(self, method: str | None = None) -> list[dict]:
        """Messages the fixture's MCP server got, in order; with `method`, only
        those. An answer to the server's own request has no method."""
        return [e for e in self.fixture if e.get("event") == "rpc" and (method is None or e["method"] == method)]

    def fixture_events(self, event: str, token: str) -> list[dict]:
        return [e for e in self.fixture if e.get("event") == event and e.get("token") == token]

    def clock(self, wall: float) -> float:
        """A wall-clock time from the fixture on the provider's clock."""
        return wall - self.epoch

    def result(self, call: Call, anywhere: bool = False) -> Result | None:
        """The result of a call, as the agent's main loop sent it back; with
        `anywhere`, also in side requests and requests the provider refused."""
        requests = [e.request for e in self.exchanges if e.request] if anywhere else self.requests
        for request in requests:
            if found := request.result(call.id):
                return found
        return None

    def step_result(self, step: int, anywhere: bool = False, track: int = 0) -> list[Result]:
        """Results of the last time a step was served, if all came back."""
        served = [s for s in self.served if s.step == step and s.track == track]
        for item in reversed(served):
            found = [self.result(call, anywhere) for call in item.calls]
            if all(found):
                return found
        return []

    def was_served(self, step: int, track: int = 0) -> bool:
        """Whether the calls of a step went to the agent."""
        return any(s.step == step and s.track == track for s in self.served)

    def invocations(self, token: str) -> list[dict]:
        return [c for c in self.fixture_calls if c["arguments"].get("token") == token]

    def tokens(self, step: int, track: int = 0) -> list[str]:
        return next((s.tokens for s in self.served if s.step == step and s.track == track), [])

    @property
    def last(self) -> LaunchRecord:
        return self.launches[-1]

    @property
    def ended(self) -> bool:
        """The agent stopped by itself: it exited, or its TUI went quiet."""
        return bool(self.launches) and self.last.termination in ("exited", "idle")

    def exit_detail(self) -> str:
        """The exit status, only as a detail: whether an agent exits 0 after an
        error depends on the output mode its recipe picks, not only on the agent."""
        if not self.ended or self.last.exit is None:
            return ""
        if self.last.exit in (-9, 137) and self.last.termination == "exited":
            gb = limits.AGENT_MEMORY / 1024**3
            return f"Killed by SIGKILL, which the suite did not send: most likely at its memory cap of {gb:g} GB."
        return f"Exit status {self.last.exit}."

    def ending(self, index: int) -> str:
        """How a launch ended, with the requests it sent, as a detail."""
        if index >= len(self.launches):
            return f"Launch {index + 1} did not start."
        record = self.launches[index]
        sent = sum(e.launch == index for e in self.exchanges)
        how = {
            "exited": f"exited with status {record.exit}",
            "deadline": f"still ran at the suite's deadline, {record.duration:.0f} s in",
            "request-limit": "reached the suite's request limit",
            "settled": "was stopped once the outcome was known",
            "killed": "was killed by the suite",
            "idle": "went quiet",
        }.get(record.termination, record.termination)
        return f"Launch {index + 1} sent {sent} request{'s' * (sent != 1)}, and {how}."

    def between(self, first: Exchange, second: Exchange) -> float:
        """Seconds from the end of one request to the start of the next."""
        return second.started - (first.finished or first.started)
