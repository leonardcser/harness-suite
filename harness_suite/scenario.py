"""What a scenario is: a script for the model, the launches it needs, and the
checks that judge the run.

A scenario sees the agent as a black box. It scripts what the model says; the
suite records what the agent sends back and what the fixture tool saw; and
each check method turns that evidence into an outcome at its level (see
`results`). One run can support several checks.
"""

import contextlib
import inspect

from . import spec
from .messages import Error, Reply
from .provider import Exchange
from .results import CheckInfo
from .script import DONE, Launch, McpServer, Step


class SpecFrontends:
    """A scenario's frontends: those of the SPEC.md area that lists its checks.
    The subject of a check decides where it runs, not what agents support."""

    def __get__(self, obj, cls) -> tuple[str, ...]:
        found = cls.checks()
        return spec.frontends()[found[0][0].id] if found else ("headless",)


class Scenario:
    """Subclasses set the fields, override `script` and `launches` as needed,
    and mark their judges with `@check`."""

    id = ""
    title = ""
    description = ""
    variants: tuple[str, ...] = ("default",)
    tools = False  # Uses the fixture tool.
    sources: tuple[str, ...] = ("mcp", "native")
    needs: tuple[str, ...] = ()  # Recipe settings, like "continue".
    # Features an agent has unless its recipe's `[unsupported]` says it lacks
    # them, with the documented reason, like "images". Where the recipe says
    # nothing, a feature that does not work fails.
    features: tuple[str, ...] = ()
    binds: tuple[str, ...] = ()  # The agent's own tools it calls, as `[native.KIND]`, like "read".
    apis: tuple[str, ...] | None = None
    frontends = SpecFrontends()  # "headless", "tui" or both.
    per_harness = False  # Run on the agent's first API only.
    window = 200_000  # The advertised context window, in tokens.
    timeout = 60.0
    max_requests = 40
    quiet = 5.0  # In the TUI, seconds without a request that end a turn.
    # Each launch's base URL has a path, `/launch/N`, as a gateway's has.
    prefixed = False
    # The provider's host resolves only through an HTTP proxy that the
    # environment names, as on a company network (`proxy`).
    proxied = False
    # The provider speaks HTTPS with a certificate the suite's own CA signed:
    # "trusted" names the CA in the environment, "untrusted" does not.
    tls: str | None = None
    # The agent may fail to offer the fixture tool: a finding, not a suite
    # error. One that lists the server's tools only after it asked the model
    # is in a race, and the case is not measured, unless `judges_wait`.
    tool_may_be_missing = False
    # The case judges whether the agent waits for its MCP server's tools, as
    # for a server that starts slowly: tools listed too late are its finding.
    judges_wait = False
    # In the TUI, the agent may never take the prompt: a finding, not a case
    # the suite could not run.
    prompt_may_be_lost = False
    # The kinds of API rule (`apis.common.Rejection`) the script makes the
    # agent likely to break, such as "tool-name" for a long tool name. A
    # request that breaks one decides the case's graded checks
    # (`engine.judge`); `request.valid` judges the other rules.
    provokes: tuple[str, ...] = ()
    # A correct agent may send no request at all, as when it must refuse a
    # prompt at once. Otherwise a pass or label needs a request.
    silent = False

    def __init__(self, variant: str, api: str, source: str, recipe, frontend: str = "headless"):
        self.variant, self.api, self.source, self.recipe = variant, api, source, recipe
        self.frontend = frontend

    @classmethod
    def checks(cls) -> list[tuple[CheckInfo, str]]:
        """Each check of the scenario, with the name of its judge method."""
        found = []
        for name in dir(cls):
            mark = getattr(inspect.getattr_static(cls, name), "check", None)
            if mark is None:
                continue
            level, id, title, description, variants, apis = mark
            info = CheckInfo(id or cls.id, level, title or cls.title, description or cls.description, variants, apis)
            found.append((info, name))
        # The check named after the scenario first.
        return sorted(found, key=lambda item: (item[0].id != cls.id, item[0].id))

    @classmethod
    def uses_tools(cls, variant: str) -> bool:
        return cls.tools

    @classmethod
    def bound(cls, variant: str) -> tuple[str, ...]:
        """The agent's own tools a variant calls: by default `binds`."""
        return cls.binds

    # What subclasses write.

    def script(self) -> list[Step]:
        return [DONE]

    def scripts(self) -> list[list[Step]]:
        """One script per track; a launch follows `Launch.track`. A launch
        that must not see the steps of the one before has a track of its own."""
        return [self.script()]

    def files(self) -> dict[str, bytes]:
        """Files to put in the workspace before the first launch."""
        return {}

    def prepare(self, sandbox) -> None:
        """Make what `files` cannot hold, such as a FIFO or a symlink, in the
        sandbox on the host before the first launch."""

    def settled(self, run) -> bool:
        """Whether the checks' outcome is known while the agent still runs, so
        the suite can stop it before its deadline. An agent stopped so counts
        as still running, as at the deadline."""
        return False

    def during(self, sandbox) -> contextlib.AbstractContextManager:
        """What runs on the host while the agent runs, such as a process that
        changes a workspace file; it stops when the context exits."""
        return contextlib.nullcontext()

    def outputs(self) -> list[str]:
        """Files to keep after the run, for the checks: `run.outputs`. A path
        is in the workspace, or with `~/` in the agent's HOME."""
        return []

    def servers(self) -> list[McpServer]:
        """The MCP servers the agent gets, with MCP tools: by default one on
        stdio that behaves as usual."""
        return [McpServer()]

    def unsupported(self) -> str | None:
        """What the agent or its recipe lacks for this case, if it lacks something."""
        return None

    def launches(self) -> list[Launch]:
        return [Launch()]

    def task(self, exchange: Exchange, model) -> Reply | Error:
        """The reply to a request of the agent's main loop."""
        return model.follow(exchange)

    def overflow(self, request) -> Error | None:
        """The error for a request over the context window: by default one
        over the advertised window."""
        return Error("context") if request.tokens > self.window else None

    def aside(self, exchange: Exchange, model) -> Reply | Error:
        """The reply to a side request, such as for a title or a summary."""
        return model.aside(exchange.request)

    def fits(self) -> bool:
        """Whether the case exists: its API, tool source and frontend can
        carry what the scenario does, whatever the agent. Subclasses add
        their own conditions. In the TUI, a launch that continues the session
        is the next prompt in the same TUI, so it cannot switch the model, API
        or settings, and the suite runs one TUI at a time."""
        if self.apis and self.api not in self.apis:
            return False
        if self.uses_tools(self.variant) and self.source not in self.sources:
            return False
        if not any(info.applies(self.variant, self.api) for info, _ in self.checks()):
            return False
        if self.frontend != "tui":
            return True
        launches = self.launches()
        switched = any(
            launch.resume and (launch.model, launch.api, launch.settings) != (before.model, before.api, before.settings)
            for before, launch in zip(launches, launches[1:], strict=False)
        )
        return not switched
