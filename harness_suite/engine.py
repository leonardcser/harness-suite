"""Run one case: one scenario variant, one agent, one API, one tool source,
one frontend. A case gives one row per check of the scenario, and one for
`request.valid`, which every case feeds."""

import contextlib
import json
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

from . import fixture
from .drivers import Headless, Tui
from .evidence import Run
from .model import Model
from .provider import Provider
from .proxy import Proxy
from .results import (
    INFO,
    RECOMMENDED,
    REQUIRED,
    CheckInfo,
    Outcome,
    conform,
    failed,
    gap,
    passed,
    skipped,
    suite_error,
    unsupported,
)
from .scenario import Scenario
from .tls import Tls

VALID = CheckInfo(
    "request.valid",
    REQUIRED,
    "Every request is one the real API accepts",
    "The fake provider checks each request against the rules of the real API:"
    " every tool call has its result and every result its call, reasoning comes"
    " back as the API requires, IDs and names are valid. A request that breaks a"
    " rule gets the real error, and fails this check. Every case feeds it. A"
    " case that provokes a rule on purpose, such as a long tool name, judges"
    " that rule in its own checks, and this check judges the other rules.",
)


def requirement(scenario: Scenario) -> str | None:
    """What the agent or its recipe lacks for the case, if anything."""
    recipe, variant, frontend = scenario.recipe, scenario.variant, scenario.frontend
    tools = scenario.uses_tools(variant)
    if frontend == "tui" and not recipe.tui:
        return "The recipe does not say how to start the TUI"
    if tools and not recipe.tools(scenario.source):
        return f"The recipe configures no {scenario.source} tool"
    if tools and scenario.source == "mcp":
        for transport in dict.fromkeys(server.transport for server in scenario.servers()):
            support = recipe.transport(transport)
            if isinstance(support, str):
                return support
            if not support:
                return f"The recipe configures no MCP server over {transport}"
    for kind in scenario.bound(variant):
        if kind not in recipe.native:
            reason = recipe.data.get("unsupported", {}).get(kind)
            return reason or f"The recipe binds no native {kind} tool"
    for feature in scenario.features:
        if reason := recipe.lacks(feature):
            return reason
    for need in scenario.needs:
        if need == "continue" and frontend == "tui":
            continue  # The next prompt goes to the same TUI.
        if need == "two-apis":
            if len(recipe.apis) < 2:
                return "The agent is configured for one API only"
            continue
        support = recipe.supports(need)
        if isinstance(support, str):
            return support
        if not support:
            return f"The recipe configures no {need} setting"
    return scenario.unsupported()


def execute(scenario: Scenario, sandbox, directory: Path) -> tuple[Run, str | None]:
    """Run the case; the evidence, and a suite error if there was one."""
    run = Run(scenario.variant, scenario.api, scenario.source, scenario.frontend, workspace=sandbox.workspace)
    paths = {
        "python": sandbox.python,
        "fixture": sandbox.fixture,
        "fixture_dir": sandbox.fixture_dir,
        "workspace": sandbox.workspace,
    }
    model = Model(scenario, run, paths)
    for name, data in scenario.files().items():
        path = sandbox.host_workspace / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    scenario.prepare(sandbox)
    with (
        mcp_servers(scenario, sandbox) as servers,
        scenario.during(sandbox),
        Provider(directory / "provider", model.respond, scenario.window, scenario.max_requests) as provider,
        front(scenario, provider.server.server_port) as front_door,
    ):
        run.exchanges, run.epoch = provider.exchanges, provider.epoch_wall
        driver = Tui if scenario.frontend == "tui" else Headless
        driver = driver(scenario, model, sandbox, provider, directory, servers, front_door)
        driver.drive()
    if isinstance(front_door, Proxy):
        run.proxied = list(front_door.seen)
    if isinstance(front_door, Tls):
        run.handshakes_failed = list(front_door.failed)
    driver.look()
    for name in scenario.outputs():
        seen = sandbox.home + name[1:] if name.startswith("~/") else f"{sandbox.workspace}/{name}"
        path = sandbox.host_path(seen)
        run.outputs[name] = path.read_bytes() if path.is_file() else None
    (directory / "fixture-calls.jsonl").write_text("".join(json.dumps(e) + "\n" for e in run.fixture))
    if provider.failure:
        return run, provider.failure
    if run.missing_at is not None and not listed_in_time(run):
        if not scenario.tool_may_be_missing or (listed_late(run) and not scenario.judges_wait):
            run.unmeasured = "The agent asked the model before it had listed its MCP server's tools."
    elif run.missing_at is not None and not scenario.tool_may_be_missing:
        return run, "The agent did not advertise the configured fixture tool."
    return run, None


def front(scenario: Scenario, provider_port: int):
    """What alone reaches the provider in this case: an HTTP proxy, a TLS
    front, or nothing."""
    if scenario.proxied:
        return Proxy(provider_port)
    if scenario.tls:
        return Tls(provider_port, trusted=scenario.tls == "trusted")
    return contextlib.nullcontext()


def listings(run: Run) -> list[float]:
    """When the agent listed its MCP servers' tools, on the run's clock."""
    return [run.clock(e["time"]) for e in run.rpc("tools/list")]


def listed_in_time(run: Run) -> bool:
    """Whether the agent had listed its MCP server's tools when it first asked
    the model without the fixture tool; True if it never spoke to a server,
    as when the recipe does not connect it."""
    spoke = run.rpc()
    return not spoke or bool(listings(run) and min(listings(run)) <= run.missing_at)


def listed_late(run: Run) -> bool:
    """Whether the agent listed its MCP server's tools, but only after it
    first asked the model without them."""
    return bool(listings(run)) and min(listings(run)) > run.missing_at


HTTP_START = 10.0  # Seconds an HTTP fixture server gets to listen.


@contextlib.contextmanager
def mcp_servers(scenario: Scenario, sandbox):
    """Set up the scenario's MCP servers; yield them as the recipe's
    templates see them. The agent starts each stdio server itself; the suite
    runs each HTTP server on the host, where the agent reaches it."""
    if scenario.source != "mcp":
        yield []
        return
    folder = sandbox.host_path(sandbox.fixture_dir)
    servers = scenario.servers()
    (folder / "servers.json").write_text(json.dumps({server.name: server.profile for server in servers}))
    processes: list[subprocess.Popen] = []
    try:
        found = []
        for server in servers:
            entry = {"server": server.name, "transport": server.transport}
            if server.transport != "stdio":
                command = [sys.executable, fixture.__file__, server.transport, str(folder), server.name]
                processes.append(subprocess.Popen(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL))
                port = listening(folder / f"{server.name}.port", processes[-1])
                entry["server_url"] = f"http://127.0.0.1:{port}/" + ("mcp" if server.transport == "http" else "sse")
            found.append(entry)
        yield found
    finally:
        for process in processes:
            process.kill()
            process.wait()


def listening(path: Path, process: subprocess.Popen) -> int:
    deadline = time.monotonic() + HTTP_START
    while time.monotonic() < deadline and process.poll() is None:
        if path.exists() and (text := path.read_text().strip()):
            return int(text)
        time.sleep(0.02)
    raise RuntimeError(f"The fixture's MCP server did not start: {path.stem}")


def broken(scenario: Scenario, run: Run, provoked: bool):
    """The first request that breaks a rule the case provokes (or, with
    `provoked` False, any other rule), with that rule's rejection."""
    for exchange in run.exchanges:
        for rejection in exchange.rejections:
            if (rejection.rule in scenario.provokes) == provoked:
                return exchange, rejection
    return None


def judge(scenario: Scenario, info: CheckInfo, name: str, run: Run) -> Outcome:
    """The check's outcome. A request that breaks a rule the case `provokes`
    decides every graded check. Unless the case is `silent`, a pass or label
    needs the agent to have reached the provider. A finding the check does
    not explain says how each launch ended."""
    if info.level != INFO and (found := broken(scenario, run, provoked=True)):
        exchange, rejection = found
        verdict = gap if info.level == RECOMMENDED else failed
        return verdict("rejected by the API", f"Request {exchange.number}: {rejection.text}")
    try:
        outcome = conform(info.level, getattr(scenario, name)(run))
    except Exception as error:
        return suite_error("suite error", f"{type(error).__name__}: {error}")
    # An agent that does nothing passes nothing.
    if outcome.result in ("pass", "label") and not scenario.silent and not run.reached:
        return skipped("The agent never reached the provider.")
    if outcome.result in ("fail", "gap") and not outcome.detail:
        outcome = replace(outcome, detail=" ".join(run.ending(i) for i in range(len(run.launches))))
    return outcome


def validity(scenario: Scenario, run: Run) -> Outcome | None:
    """`request.valid`: every rule the case does not provoke holds."""
    if not any(e.request or e.rejections for e in run.exchanges):
        return None
    found = broken(scenario, run, provoked=False)
    if found is None:
        return passed()
    exchange, rejection = found
    where = f"{scenario.id}/{scenario.variant}"
    return failed("invalid request", f"{where}, request {exchange.number} ({exchange.api}): {rejection.text}")


def case(scenario: Scenario, sandbox_factory, directory: Path) -> list[dict]:
    """Run a case; its rows. A sandbox factory that is a string says why the
    agent cannot run at all."""
    started = time.monotonic()
    checks = [(info, name) for info, name in scenario.checks() if info.applies(scenario.variant, scenario.api)]
    if isinstance(sandbox_factory, str):
        return [row(info, scenario, skipped(sandbox_factory), 0) for info, _ in checks]
    if lacking := requirement(scenario):
        return [row(info, scenario, unsupported(detail=lacking), 0) for info, _ in checks]
    directory.mkdir(parents=True, exist_ok=True)
    outcomes: list[tuple[CheckInfo, Outcome]]
    sandbox = sandbox_factory()
    try:
        run, error = execute(scenario, sandbox, directory)
    except Exception as caught:
        run, error = None, f"{type(caught).__name__}: {caught}"
    finally:
        sandbox.close()
    if error:
        outcomes = [(info, suite_error("suite error", error)) for info, _ in checks]
    elif run.unmeasured:
        outcomes = [(info, skipped(run.unmeasured)) for info, _ in checks]
    else:
        outcomes = [(info, judge(scenario, info, name, run)) for info, name in checks]
        if valid := validity(scenario, run):
            outcomes.append((VALID, valid))
    seconds = time.monotonic() - started
    rows = [row(info, scenario, outcome, seconds) for info, outcome in outcomes]
    for item in rows:
        item["artifact"] = str(directory)
    (directory / "result.json").write_text(json.dumps(rows, indent=2))
    return rows


def row(info: CheckInfo, scenario: Scenario, outcome: Outcome, seconds: float) -> dict:
    return {
        "harness": scenario.recipe.name,
        "version": scenario.recipe.version,
        "api": scenario.api,
        "tools": scenario.source,
        "uses_tools": scenario.uses_tools(scenario.variant),
        "frontend": scenario.frontend,
        "case": f"{scenario.id}/{scenario.variant}",
        "check": info.id,
        # A check that every case feeds has one variant; `case` says where.
        "variant": "default" if info is VALID else scenario.variant,
        "level": info.level,
        "result": outcome.result,
        "summary": outcome.summary,
        "detail": outcome.detail,
        "seconds": round(seconds, 1),
    }
