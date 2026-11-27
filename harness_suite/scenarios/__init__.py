"""Every scenario, one module per area. A new scenario is a new class here."""

from ..engine import VALID
from ..results import CheckInfo
from ..scenario import Scenario
from . import (
    cache,
    cli,
    context,
    errors,
    files,
    mcp,
    media,
    reasoning,
    request,
    session,
    settings,
    shell,
    stream,
    tools,
    workspace,
)

# In the order of the spec's areas.
MODULES = (
    request, stream, errors, cache, reasoning, tools, shell, files, mcp, media, context, session, settings,
    workspace, cli,
)  # fmt: skip


def all_scenarios() -> list[type[Scenario]]:
    found = []
    for module in MODULES:
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, Scenario) and value.__module__ == module.__name__:
                if value.id and value not in found:
                    found.append(value)
    return found


def all_checks() -> list[tuple[CheckInfo, type[Scenario]]]:
    """Every check, with the scenario whose runs it judges; `request.valid`
    first, which every scenario feeds."""
    return [(VALID, Scenario)] + [(info, s) for s in all_scenarios() for info, _ in s.checks()]


def find(selector: str) -> list[type[Scenario]]:
    """Scenarios whose id, or the id of one of their checks, equals or starts
    with `selector` (an area like "tools")."""

    def matches(name: str) -> bool:
        return name == selector or name.startswith(selector + ".")

    return [s for s in all_scenarios() if matches(s.id) or any(matches(info.id) for info, _ in s.checks())]
