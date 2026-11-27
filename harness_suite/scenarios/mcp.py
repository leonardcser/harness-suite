"""MCP: the protocol between the agent and its tool servers.

The fixture's server logs every message it gets, and a profile per case
changes how it behaves (see `fixture.PROFILE`). Most checks ask whether the
fixture tool still works when the server uses a part of the protocol that
servers in the wild use; the rest record which optional parts the agent
supports.
"""

import json
import time
from dataclasses import replace

from ..fixture import LISTED_AGAIN, marker
from ..results import (
    INFO,
    OPTIONAL,
    RECOMMENDED,
    REQUIRED,
    check,
    demoted,
    failed,
    gap,
    label,
    passed,
    skipped,
    unsupported,
)
from ..scenario import Scenario
from ..script import DONE, McpServer, Use, use
from .tools import delivered


class Server(Scenario):
    """One fixture call over MCP, with the server set up by `profile`."""

    tools = True
    sources = ("mcp",)
    profile: dict = {}  # How the server behaves: see `fixture.PROFILE`.
    kind = "text"
    tool_may_be_missing = True  # An agent that drops the server is a finding.

    def servers(self):
        return [McpServer(profile=self.profile)]

    def script(self):
        return [use(self.kind), DONE]

    def works(self, run, required: bool = False):
        """The call ran once and its result came back."""
        if run.missing_at is not None:
            missing = failed("tool not offered", "The agent never offered the server's tool to the model.")
            return missing if required else demoted(missing)
        if problem := delivered(run, 0, [self.kind]):
            return problem if required else demoted(problem)
        return passed()


class Handshake(Server):
    id = "mcp.handshake"
    title = "The agent opens the MCP session as the protocol requires"
    description = (
        "The agent must send `initialize`, wait for the answer, send the"
        " `notifications/initialized` notification, and only then list or call"
        " tools. A server may ignore anything that comes before it."
    )

    @check(REQUIRED)
    def ordered(self, run):
        # Newer protocol versions may probe the server first, as with
        # `server/discover`; what matters is that no tool request comes early.
        methods = [e["method"] for e in run.rpc() if e["method"]]
        if "initialize" not in methods:
            return failed("no initialize", "The agent never sent initialize.")
        if "notifications/initialized" not in methods:
            return failed("no initialized", "The agent never sent notifications/initialized.")
        early = [m for m in methods[: methods.index("notifications/initialized")] if m.startswith("tools/")]
        if early:
            return failed("too early", f"{early[0]} came before notifications/initialized.")
        return self.works(run, required=True)

    @check(
        INFO,
        id="mcp.server-env",
        title="Whether an MCP server gets the provider's API key",
        description=(
            "The agent starts the stdio server, and the server notes which of its"
            " environment variables hold the API key the recipe gave the agent. An"
            " agent that passes its whole environment on hands the key to every"
            " server it starts; one that passes a small default set, as the MCP"
            " SDKs do, does not."
        ),
    )
    def environment(self, run):
        found = [e for e in run.fixture if e.get("event") == "environment"]
        if not found:
            return skipped("The server never started.")
        held = sorted({name for e in found for name in e.get("key_in", [])})
        return label("key passed on", ", ".join(held)) if held else label("key kept back")


class Version(Server):
    id = "mcp.version"
    title = "A server on an older protocol version still works"
    description = (
        "The server answers `initialize` with the first protocol version,"
        " 2024-11-05, whatever the agent asked for, as many published servers"
        " do. The agent should go on with that version and use the tool."
    )
    profile = {"version": "2024-11-05"}

    @check(RECOMMENDED)
    def used(self, run):
        return self.works(run)


class Startup(Server):
    id = "mcp.startup"
    title = "A server that starts slowly or logs a lot still works"
    description = (
        "The server takes 8 s to answer `initialize`, as a server that installs"
        " its packages first does (`slow`), or writes 256 KB of logs to stderr as"
        " it starts (`noisy`). The agent should wait for it and drain its stderr."
    )
    variants = ("slow", "noisy")
    timeout = 90.0

    @property
    def judges_wait(self):
        return self.variant == "slow"

    @property
    def profile(self):
        return {"delay": 8.0} if self.variant == "slow" else {"stderr": 256 * 1024}

    @check(RECOMMENDED)
    def used(self, run):
        return self.works(run)


class Framing(Server):
    id = "mcp.framing"
    title = "The agent reads stdio messages however the server writes them"
    description = (
        "Over stdio, the server writes each message in 16-byte pieces, so that one"
        " message takes many reads (`split`), or first writes a line that is not"
        " JSON, as a server that announces itself on stdout does (`banner`). The"
        " protocol has messages end at a newline and forbids other output, but"
        " servers in the wild do both. The tool should still work."
    )
    variants = ("split", "banner")

    @property
    def profile(self):
        return {"framing": self.variant}

    @check(RECOMMENDED)
    def used(self, run):
        return self.works(run)


class Mute(Server):
    id = "mcp.mute"
    title = "A server that never answers does not block the agent"
    description = (
        "The server reads `initialize` and never answers, as a hung or broken"
        " server does. The agent should give up on it in time and do the task"
        " without its tools. The task needs no tool; the suite waits 120 s."
    )
    profile = {"mute": True}
    timeout = 120.0

    def script(self):
        return [DONE]

    @check(RECOMMENDED)
    def went_on(self, run):
        if not run.tasks:
            return gap("blocked", f"The agent sent no request in {self.timeout:.0f} s. {run.exit_detail()}".strip())
        waited = run.tasks[0].started - run.launches[0].started
        if not run.finished_script:
            return gap("stopped", f"The agent stopped after {waited:.0f} s.")
        return passed("went on", f"The first task request came {waited:.0f} s after the start.")


class Restart(Server):
    id = "mcp.restart"
    title = "A server that exits between calls"
    description = (
        "The server exits after its first call, as a server that crashes does."
        " Then the model calls the tool again. The agent should start the server"
        " again or give the model an error result, and finish the task."
    )
    profile = {"exit_after": 1}

    def script(self):
        return [use("text"), use("text"), DONE]

    @check(RECOMMENDED)
    def recovered(self, run):
        if run.missing_at is not None:
            return gap("tool not offered")
        if problem := delivered(run, 0, ["text"]):
            return demoted(problem)
        results = run.step_result(1)
        if not results:
            return gap("no result", f"The second call got no result. {run.exit_detail()}".strip())
        if not run.finished_script:
            return gap("stopped", "The agent did not go on after the second call.")
        return passed()


class Notifications(Server):
    id = "mcp.notifications"
    title = "Notifications from the server do not break a call"
    description = (
        "During the call, the server sends a log message (`notifications/message`,"
        " with the `logging` capability declared) and, if the call carries a"
        " progress token, three `notifications/progress`, as many servers do. The"
        " call must still work. The detail says whether the agent set a log level"
        " and asked for progress."
    )
    profile = {"logs": True, "progress": True}

    @check(RECOMMENDED)
    def unaffected(self, run):
        outcome = self.works(run)
        levels = run.rpc("logging/setLevel")
        if outcome.result != "pass":
            return outcome
        level = (levels[0].get("params") or {}).get("level") if levels else None
        calls = run.rpc("tools/call")
        asked = calls and "progressToken" in ((calls[0].get("params") or {}).get("_meta") or {})
        said = f"The agent set the level to {level}" if level else "The agent set no level"
        return passed(detail=f"{said}, and {'asked' if asked else 'did not ask'} for progress.")


NATIVE_NOTE, NATIVE_MARK = "notes/suite-native.txt", "SUITE_NATIVE_READ"


class NativeNames(Server):
    id = "mcp.native-names"
    title = "An MCP tool with the name of the agent's own tool"
    description = (
        "Next to the fixture, the server lists a tool with the name of each of the"
        " agent's own tools, as servers that pick common names like `read` or"
        " `bash` do. The model reads a file with the agent's own read tool, then"
        " calls the fixture. The agent must offer no name twice, which Anthropic's"
        " API rejects, and the read must run the agent's own tool, not the"
        " server's."
    )
    binds = ("read",)
    provokes = ("tool-name",)

    @property
    def profile(self):
        # A sub-agent can be several calls, each a tool.
        specs = [one for spec in self.recipe.native.values() for one in (spec if isinstance(spec, list) else [spec])]
        return {"shadows": list(dict.fromkeys(spec["name"] for spec in specs))}

    def files(self):
        return {NATIVE_NOTE: f"{NATIVE_MARK}\n".encode()}

    def script(self):
        return [use(Use("read", value=NATIVE_NOTE)), use("text"), DONE]

    @check(REQUIRED)
    def apart(self, run):
        for request in run.requests:
            names = [tool.name for tool in request.tools]
            if twice := sorted({name for name in names if names.count(name) > 1}):
                return failed("same name twice", f"A request offers {', '.join(twice)} twice.")
        shadows = self.profile["shadows"]
        if ran := [c["params"]["name"] for c in run.rpc("tools/call") if c["params"].get("name") in shadows]:
            return failed("server's tool ran", f"The model's call of {ran[0]} ran the MCP server's tool.")
        results = run.step_result(0)
        if not results or NATIVE_MARK not in results[0].text:
            return failed("not read", f"The read's result: {results[0].text[:160]!r}" if results else "No result.")
        if problem := delivered(run, 1, ["text"]):
            return problem
        return passed()


class Pagination(Server):
    id = "mcp.pagination"
    title = "Tools on a later page of `tools/list` are found"
    description = (
        "The server lists its tools two at a time, with a `nextCursor`, and the"
        " fixture is on the third page. The agent should follow the cursor and"
        " offer the fixture to the model."
    )
    profile = {"page": 2, "decoys": 5}

    @check(RECOMMENDED)
    def found(self, run):
        cursors = [e for e in run.rpc("tools/list") if (e.get("params") or {}).get("cursor")]
        outcome = self.works(run)
        if outcome.result != "pass":
            return gap("not found", "Only the first page was read.") if not cursors else outcome
        return passed(detail=f"{len(cursors)} later pages read.")


DECOYS = 150


class ManyTools(Server):
    id = "mcp.many-tools"
    title = "A server with many tools still works"
    description = (
        f"The server lists {DECOYS} other tools before the fixture, as a user with"
        " several large servers has. The agent should still offer the fixture to the"
        " model, at once or through its own tool search, in requests the API takes:"
        " OpenAI's Chat Completions takes at most 128 tools. The detail shows how"
        " many tools the first request carries."
    )
    profile = {"decoys": DECOYS}
    provokes = ("tool-count",)

    @check(RECOMMENDED)
    def used(self, run):
        outcome = self.works(run)
        if outcome.result != "pass" or not run.tasks:
            return outcome
        sent = len(run.tasks[0].request.tools)
        return passed(detail=f"{sent} tools in the first request.")


RELIST_WAIT = 10.0  # Seconds the model waits for the agent to list the tools again.


class ListChanged(Server):
    id = "mcp.list-changed"
    title = "The agent reloads the tools when the server says they changed"
    description = (
        "After the first call, the server changes its tool's description and sends"
        " `notifications/tools/list_changed`, as servers do when their tools"
        " depend on state. The model answers the first result only once the agent"
        f" has listed the tools again, or after {RELIST_WAIT:g} s, and calls the"
        " tool again. A later request must show the new description, in the tools"
        " or in what the agent's own tool search returned, and the second call"
        " must work."
    )
    profile = {"changes": True}
    features = ("mcp-list-changed",)

    def script(self):
        return [use("text"), use("text"), DONE]

    def task(self, exchange, model):
        # The agent reloads its tools as it sends the first result: wait for
        # the reload, and a second for the agent to take it in.
        first = model.run.served[0].calls[0].id if model.run.served else None
        if first and len(model.run.served) == 1 and exchange.request.result(first):
            deadline = time.monotonic() + RELIST_WAIT
            while len(model.run.rpc("tools/list")) < 2 and time.monotonic() < deadline:
                time.sleep(0.05)
            time.sleep(1.0)
        return super().task(exchange, model)

    @check(OPTIONAL)
    def reloaded(self, run):
        if not run.was_served(1):
            return skipped("The second call was never served, so no request came after the change.")
        first = next((e.number for e in run.tasks if e.request.result(run.served[0].calls[0].id)), None)
        later = [e for e in run.tasks if first is not None and e.number >= first]
        if not any(LISTED_AGAIN in e.request.everything for e in later):
            relisted = len(run.rpc("tools/list")) > 1
            why = "It listed the tools again" if relisted else "It never listed the tools again"
            return failed("not reloaded", f"{why}, and no later request shows the new description.")
        return delivered(run, 1, ["text"]) or passed()


class ProtocolError(Server):
    id = "mcp.protocol-error"
    title = "A call the server answers with a protocol error gets an error result"
    description = (
        "The server answers `tools/call` with a JSON-RPC error instead of a result,"
        " as a server does when it fails inside. The agent must give the model an"
        " error result for the call and go on."
    )
    kind = "protocol-error"

    @check(REQUIRED)
    def answered(self, run):
        if not run.served:
            return skipped("The call was never served.")
        results = run.step_result(0)
        if not results:
            return failed("no result", f"The call never got a result. {run.exit_detail()}".strip())
        if not run.finished_script:
            return failed("stopped", "The agent did not go on after the error.")
        return passed(detail=f"Result: {results[0].text[:120]!r}")


class Transport(Server):
    id = "mcp.transport"
    title = "The agent reaches an MCP server over HTTP"
    description = (
        "The fixture's server listens on a local port instead of stdio, over"
        " Streamable HTTP (`http`), the current transport, or over HTTP with"
        " server-sent events (`sse`), the older one that many servers still use."
        " One call must reach the server and its result come back."
    )
    variants = ("http", "sse")

    def servers(self):
        return [McpServer(transport=self.variant)]

    @check(OPTIONAL)
    def works_over_http(self, run):
        # The recipe configures the transport, so a tool never offered fails.
        if run.missing_at is not None:
            return failed(
                "tool not offered", f"The agent never offered the server's tool to the model. {self.tried(run)}"
            )
        return self.works(run, required=True)

    def tried(self, run) -> str:
        """What the fixture's server saw of the agent."""
        events = [e["event"] for e in run.fixture]
        if not events:
            return "It never reached the server."
        if self.variant == "sse" and "sse-stream" not in events:
            return "It sent POST requests, as with Streamable HTTP, and never opened the SSE stream."
        methods = [e["method"] for e in run.rpc() if e.get("method")]
        return f"The server got: {', '.join(dict.fromkeys(methods)) or ', '.join(dict.fromkeys(events))}."


class SameNames(Server):
    id = "mcp.same-names"
    title = "Two MCP servers with the same tool names"
    description = (
        "Two servers, `suite` and `suite_b`, both list a tool named"
        " `suite_fixture`, as two copies of one server, or two servers that"
        " pick common names, would. The model calls the tool of each server once,"
        " found by the server name in its description. The agent must offer both"
        " under names that differ and are valid, and send each call to its own"
        " server."
    )
    NAMES = ("suite", "suite_b")
    provokes = ("tool-name",)

    def servers(self):
        return [McpServer(name) for name in self.NAMES]

    def script(self):
        return [use(Use("text", server=name)) for name in self.NAMES] + [DONE]

    @check(REQUIRED)
    def routed(self, run):
        # In a tool, or in what the agent's own tool showed of the servers.
        seen = "\n".join(t.description for r in run.requests for t in r.tools)
        seen += "\n".join(r.visible for r in run.requests)
        if lost := [name for name in self.NAMES if f"Server {name}." not in seen]:
            return failed("one hidden", f"The model never saw the tool of {', '.join(lost)}.")
        for step, name in enumerate(self.NAMES):
            if problem := delivered(run, step, ["text"]):
                return problem
            reached = {e.get("server") for token in run.tokens(step) for e in run.invocations(token)}
            if reached != {name}:
                got = ", ".join(sorted(str(r) for r in reached)) or "no server"
                return failed("wrong server", f"The call for {name} ran on {got}.")
        return passed()


class SessionExpired(Server):
    id = "mcp.session-expired"
    title = "The agent opens a new MCP session when the server forgot the old one"
    description = (
        "Over Streamable HTTP, the server forgets the session after the first"
        " call, as a server that restarted does, and answers its ID with a 404."
        " The protocol says the client must then start a new session with"
        " `initialize`. The model's second call must work."
    )

    def servers(self):
        return [McpServer(transport="http", profile={"forget_after": 1})]

    def script(self):
        return [use("text"), use("text"), DONE]

    @check(RECOMMENDED)
    def reopened(self, run):
        if run.missing_at is not None:
            return skipped("The agent never offered the server's tool, so it does not reach this HTTP server.")
        if not any(e.get("event") == "http-session-forgotten" for e in run.fixture):
            return skipped("The first call never finished, so the server kept its session.")
        if problem := delivered(run, 1, ["text"]):
            opened = len(run.rpc("initialize"))
            return demoted(replace(problem, detail=f"{problem.detail} The agent opened {opened} session(s)."))
        return passed()


MCP_RULE = "SUITE_MCP_RULE: call suite_fixture only with the token you were given."


class Instructions(Server):
    id = "mcp.instructions"
    title = "Where the server's instructions reach the model"
    description = (
        "The server's answer to `initialize` holds `instructions`, which the"
        " protocol says a client may give the model to explain the server's"
        " tools. The label shows where they are in the first request the agent"
        " sent once it had listed the server's tools: the system prompt, the"
        " description of a tool or a group of tools, a message, or left out."
    )
    profile = {"instructions": MCP_RULE}

    @check(INFO)
    def placed(self, run):
        listed = [run.clock(e["time"]) for e in run.rpc("tools/list")]
        if not listed:
            return skipped("The agent never listed the server's tools.")
        first = next((e for e in run.tasks if e.started >= min(listed)), None)
        if first is None:
            return skipped("The agent sent no task request after it listed the server's tools.")
        found = first.request.place(MCP_RULE)
        if found is None:
            return label("left out")
        if found.startswith("message"):
            return label("message", f"The first request holds them in {found}.")
        return label(found)


class UnusableTool(Server):
    id = "mcp.unusable-tool"
    title = "A server tool the model APIs cannot take does not break requests"
    description = (
        "Next to the fixture, the server lists a tool that no model API takes as"
        " it is: named `suite.odd/tool name!` (`bad-name`), with an input schema"
        " whose root is a string (`not-object`), or with no input schema at all"
        " (`no-schema`). A real API rejects every request that holds such a tool."
        " The agent should fix or skip the tool, so that its requests stay valid"
        " and the fixture still works. The last two break the MCP spec, which"
        " requires an object schema, so an agent may also refuse the whole server."
    )
    variants = ("bad-name", "not-object", "no-schema")
    provokes = ("tool-name", "tool-schema")

    @property
    def profile(self):
        if self.variant == "bad-name":
            return {"odd": True}
        return {"broken": "not-object" if self.variant == "not-object" else "missing"}

    @check(RECOMMENDED)
    def valid(self, run):
        if self.variant != "bad-name" and run.missing_at is not None:
            if not run.rpc("tools/list"):
                return skipped("The agent never listed the server's tools.")
            return passed("server refused", "The agent listed the tools, then offered none of them.")
        return self.works(run)


class Schema(Server):
    id = "mcp.schema"
    title = "A tool with a rich JSON schema still works"
    description = (
        "The fixture's input schema uses `$defs`, `$ref`, `anyOf` and enums in"
        " arrays, as schemas that libraries generate do. The agent must pass on a"
        " schema the API accepts, and the tool must still work. A pass says whether"
        " the agent sent the schema as is, with its references resolved, or"
        " simplified."
    )
    profile = {"rich": True}
    provokes = ("tool-schema",)

    @check(RECOMMENDED)
    def works_rich(self, run):
        outcome = self.works(run)
        if outcome.result != "pass":
            return outcome
        offered = [t for e in run.tasks for t in e.request.tools if t.name.endswith("suite_fixture")]
        text = json.dumps(offered[0].schema) if offered else ""
        if "$ref" in text:
            return passed("as is")
        return passed("references resolved" if "options" in text else "simplified")


class Structured(Server):
    id = "mcp.structured"
    title = "What reaches the model from a structured tool result"
    description = (
        "The fixture declares an output schema, and each result has a text block"
        " and `structuredContent` with other words. The label shows which of the"
        " two the model saw."
    )
    profile = {"structured": True}

    @check(INFO)
    def seen(self, run):
        results = run.step_result(0)
        if not results:
            return skipped("The result never came back; mcp.handshake and tools.round-trip judge that.")
        token = run.tokens(0)[0]
        text, structured = marker(token) in results[0].text, f"SUITE_STRUCT_{token}" in results[0].text
        return label(
            {(True, True): "both", (True, False): "text", (False, True): "structured"}.get(
                (text, structured), "neither"
            )
        )


NOT_ASKED = "The call never ran, so the server never sent its request."


class Ask(Server):
    """The server sends the agent a request while a call runs."""

    method = ""

    @property
    def profile(self):
        return {"asks": [self.method]}

    def script(self):
        return [use(Use("slow", seconds=1)), DONE]

    @property
    def kind(self):
        return "slow"

    def answer(self, run):
        """The agent's answer: "skipped" if it did not declare the capability,
        "not asked" if the server never got to send its request."""
        if any(e.get("event") == "rpc-skip" for e in run.fixture):
            return "skipped"
        if not any(e.get("event") == "rpc-out" for e in run.fixture):
            return "not asked"
        return next((e for e in run.rpc() if e["method"] is None), None)


class Capability(Ask):
    """The server asks for something only an agent that declares `capability`
    offers; one that does not declare it does not support it."""

    capability = ""

    def judged(self, run):
        """The outcome for a missing or absent answer, or the answer."""
        answer = self.answer(run)
        if answer == "not asked":
            return skipped(NOT_ASKED)
        if answer == "skipped":
            return unsupported(detail=f"The agent does not declare the `{self.capability}` capability.")
        if answer is None:
            return failed("no answer", f"The agent declared {self.capability} but never answered.")
        return answer


class Ping(Ask):
    id = "mcp.ping"
    title = "The agent answers the server's ping"
    description = (
        "While a call runs, the server sends `ping`, as servers do to see the client"
        " is alive. The protocol requires an answer, and the call must still finish."
    )
    method = "ping"

    @check(RECOMMENDED)
    def answered(self, run):
        answer = self.answer(run)
        if answer == "not asked":
            return skipped(NOT_ASKED)
        if answer is None:
            return gap("no answer", "The agent never answered the ping.")
        if answer["answer"] != "result":
            return gap("error", f"The agent answered the ping with an error: {answer['result'][:160]}")
        return self.works(run)


class Roots(Capability):
    id = "mcp.roots"
    title = "The agent tells the server its workspace"
    description = (
        "If the agent declares the `roots` capability, the server asks for the"
        " roots while a call runs. The answer should list the workspace."
    )
    method, capability = "roots/list", "roots"

    @check(OPTIONAL)
    def listed(self, run):
        answer = self.judged(run)
        if not isinstance(answer, dict):
            return answer
        if run.workspace not in answer["result"]:
            return failed("other roots", answer["result"][:160])
        return passed()


class Elicitation(Capability):
    id = "mcp.elicitation"
    title = "The agent answers the server's question to the user"
    description = (
        "If the agent declares `elicitation`, the server asks the user for a value"
        " while a call runs. Headless, no user is there: the agent should still"
        " answer, for example by declining, so the call does not hang."
    )
    method, capability = "elicitation/create", "elicitation"

    @check(OPTIONAL)
    def answered(self, run):
        answer = self.judged(run)
        return passed(detail=answer["result"][:160]) if isinstance(answer, dict) else answer


class Sampling(Capability):
    id = "mcp.sampling"
    title = "The agent serves the server's request for a completion"
    description = (
        "If the agent declares `sampling`, the server asks it for a model"
        " completion while a call runs. The agent should answer, with a"
        " completion or a refusal, so the call does not hang."
    )
    method, capability = "sampling/createMessage", "sampling"

    @check(OPTIONAL)
    def answered(self, run):
        answer = self.judged(run)
        return passed(detail=answer["result"][:160]) if isinstance(answer, dict) else answer
