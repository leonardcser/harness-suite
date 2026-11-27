"""The scripted model's lines: the steps it says, and the launches of the agent."""

from collections.abc import Callable
from dataclasses import dataclass, field

from .messages import Error, Stream

TASK = "SUITE_TASK"
PROMPT = f"{TASK}: Follow the instructions of the test model. The test model scripts every step."
AUX = "SUITE_AUX_REPLY"  # Starts every reply to a side request.


@dataclass(frozen=True)
class Use:
    """A call of the fixture tool. The suite fills in the tool name."""

    kind: str
    size: int | None = None
    style: str | None = None
    value: str | None = None
    seconds: float | None = None
    arguments: str | None = None  # Raw argument JSON, sent as is.
    broken: str | None = None  # Break the final arguments: "half", "unclosed" or "list".
    name: str | None = None  # Call a tool by this name instead.
    command: str | None = None  # With the shell tool: run this command, not the fixture.
    heredoc: bool = False  # With the shell tool: give the fixture its arguments as a heredoc.
    # The agent's own file tools, kinds "read", "write" and "edit": `value` is
    # the path in the workspace; a write gives `content`, an edit `old` and `new`.
    content: str | None = None
    old: str | None = None
    new: str | None = None
    stage: int = 0  # Kind "subagent": which of the recipe's calls; `content` is the sub-agent's task.
    token: str | None = None  # The same token makes calls identical.
    server: str | None = None  # Over MCP, the server whose tool to call; the first by default.


@dataclass(frozen=True)
class McpServer:
    """An MCP server of the fixture that the agent gets for a case."""

    name: str = "suite"
    transport: str = "stdio"  # Or "http", Streamable HTTP, or "sse", the older HTTP with SSE.
    profile: dict = field(default_factory=dict)  # How it behaves: see `fixture.PROFILE`.


@dataclass(frozen=True)
class Step:
    """One thing the model says. A step with calls waits for their results."""

    text: str = ""
    uses: tuple[Use, ...] = ()
    thinking: str = ""
    thinking_form: str = "shown"  # See Reply.thinking_form.
    stream: Stream = field(default_factory=Stream)
    stop: str | None = None
    error: Error | None = None
    times: int = 1  # For errors: how many requests get it.
    call_ids: tuple[str, ...] | None = None  # Fixed call IDs, to reuse one.

    @property
    def faulty(self) -> bool:
        """The stream does not deliver a complete reply: it breaks, or it ends
        without its finish events or with a stop reason that says the reply is
        not whole."""
        s = self.stream
        return bool(
            s.cut_after is not None
            or s.error_after is not None
            or s.corrupt is not None
            or s.drop_terminal
            or self.stop in INCOMPLETE
        )


# Stop reasons of a reply that is not whole: the output limit, a failed
# stream (Chat's `error`), or Anthropic's `model_context_window_exceeded`.
INCOMPLETE = ("length", "error", "window")


def say(text: str, **options) -> Step:
    return Step(text=text, **options)


def use(*uses: Use | str, text: str = "", **options) -> Step:
    return Step(text=text, uses=tuple(Use(u) if isinstance(u, str) else u for u in uses), **options)


def slow_started(run) -> bool:
    """Whether the fixture has started a slow call: a trigger for `kill_when`."""
    return any(call["arguments"].get("kind") == "slow" for call in run.fixture_calls)


def fail_with(kind: str, times: int = 1, **options) -> Step:
    return Step(error=Error(kind, **options), times=times)


DONE = say("SUITE_DONE: the scripted task is complete.")


@dataclass(frozen=True)
class Launch:
    """One start of the agent, or in the TUI one prompt. A later launch with
    `resume` continues the session."""

    prompt: str = PROMPT
    settings: dict = field(default_factory=dict)  # Recipe settings, like effort.
    model: str | None = None
    api: str | None = None
    resume: bool = False
    # Headless: the suite kills the agent with SIGKILL once `kill_when(run)`
    # is true, as a crash would end it.
    kill_when: Callable | None = None
    timeout: float | None = None
    track: int = 0  # The script it follows: see `Scenario.scripts`.
    # Headless: words after the recipe's arguments. The recipe's command may
    # start with an interpreter, which would read words put before them.
    options: tuple[str, ...] = ()
    # Add a line no format parses to the first file the recipe writes, its
    # configuration.
    broken_file: bool = False
    # Headless: stdin is a pipe the suite holds open, as a script's often is,
    # instead of an empty one. A recipe that sends the prompt on stdin then
    # sends it as the last argument.
    open_stdin: bool = False
