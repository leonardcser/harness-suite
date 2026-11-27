"""The scripted model: it tells the agent's main loop from its side requests,
follows the scenario's script in the main loop, and answers side requests as a
real model would."""

import itertools
import json
import re
import shlex
from dataclasses import replace

from .evidence import Run, Served
from .fixture import DESCRIPTION, marker
from .messages import Call, Error, Reply, Request, Result
from .provider import Exchange
from .recipe import MODELS, render
from .script import AUX, TASK, Step, Use

ASK_SUMMARY = re.compile(
    r"\b(?:summari[sz]e\b|(?:produce|write|create|generate|provide|output|return|update)\b[^.\n]{0,60}?"
    r"\b(?:summary|checkpoint|handoff|compaction))",
    re.IGNORECASE,
)
# A turn that reuses the conversation for something else, such as the
# session's title, and says the tools are off for it.
ASK_ASIDE = re.compile(
    r"<title-request>|\bname this (?:session|conversation)\b"
    r"|\b(?:do not|don't|never) (?:emit|make|send) (?:any )?tool calls\b|\btools (?:are )?not available\b",
    re.IGNORECASE,
)
# A clause that tells the model not to do something, such as "do not summarize".
NEGATED = re.compile(r"\b(?:do not|don't|never)\b[^.\n]*", re.IGNORECASE)
HEADING = re.compile(r"^#{1,4} [^\n\[]+$", re.MULTILINE)
OWN_TOOLS = ("read", "write", "edit", "subagent")  # The agent's own, bound as `[native.KIND]`.
ASIDE_TAG = re.compile(AUX + r"_\d+")
FACT = re.compile(r"SUITE_(?:FACT_[A-Z0-9_]+=[\w-]+|RESULT_\w+)")


def aside_tag(text: str) -> str | None:
    """The tag that starts a reply of `Model.aside`, if the text has one."""
    found = ASIDE_TAG.search(text)
    return found.group() if found else None


class Track:
    """Where the model is in one script. Launches on one track share it, so a
    launch that continues the session goes on where the one before stopped;
    agents that run at once each follow their own. Each agent also loads a
    deferred fixture tool for itself, so each track has its own `bootstrap`
    calls, by MCP server."""

    def __init__(self, number: int, steps: list[Step], scenario_id: str, bootstrap: dict[str, list[dict]]):
        if not steps or steps[-1].uses or steps[-1].error:
            raise ValueError(f"{scenario_id}: a script must end with a text step")
        self.number, self.steps, self.bootstrap = number, steps, bootstrap
        self.position = 0
        self.repeats = 0
        self.pending: Served | None = None
        self.finished = False  # The last step was served.
        self.cut = ""  # The text of the last step served with a faulty stream.

    @property
    def done(self) -> bool:
        """The script's last step was served and no call is open."""
        return self.finished and self.pending is None


class Model:
    def __init__(self, scenario, run: Run, paths: dict):
        self.scenario, self.run, self.paths = scenario, run, paths
        self.recipe, self.source = scenario.recipe, scenario.source
        self.servers = [server.name for server in scenario.servers()] if self.source == "mcp" else []
        self.tracks = [
            Track(n, steps, scenario.id, {name: self.recipe.bootstrap(name) for name in self.servers})
            for n, steps in enumerate(scenario.scripts())
        ]
        self.launch_tracks = [launch.track for launch in scenario.launches()]
        self.models = set(MODELS)  # The engine adds the models that launches pick.
        self.routed: dict[str, str] = {}  # By MCP server.
        self.issued: set[str] = set()  # Every call the model sent.
        self.asides = 0  # Side requests answered: each reply has its own tag.

    def track(self, launch: int) -> Track:
        """The track a launch follows."""
        return self.tracks[self.launch_tracks[launch] if launch < len(self.launch_tracks) else 0]

    def respond(self, exchange: Exchange) -> Reply | Error:
        request = exchange.request
        if (overflow := self.scenario.overflow(request)) is not None:
            exchange.kind = "over-window"
            return overflow
        task, why = self.classify(request, self.track(exchange.launch))
        exchange.notes["why"] = why
        if not task:
            exchange.kind = "aux"
            return self.scenario.aside(exchange, self)
        exchange.kind = "task"
        return self.scenario.task(exchange, self)

    def classify(self, request: Request, track: Track) -> tuple[bool, str]:
        """Whether a request is the agent's main loop, and why."""
        if request.model not in self.models:
            return False, f"model {request.model}"
        if not request.tools:
            return False, "no tools"
        if request.parameters.get("tool_choice") in ("none", {"type": "none"}):
            return False, "tool_choice none"
        last = request.last
        if last is None:
            return False, "no messages"
        if last.role == "user":
            # A request for a summary is a side request, as a real model would
            # read it, even when it quotes the user's prompt. The user's prompt,
            # or a checkpoint that carries an earlier summary, hands over the task.
            if ASK_SUMMARY.search(NEGATED.sub("", last.text)):
                return False, "asks for a summary"
            if ASK_ASIDE.search(last.text):
                return False, "asks for something else, without tools"
            if TASK in last.text:
                return True, "the task prompt"
            if AUX in last.text:
                return True, "a checkpoint with a summary"
        # The model's reply broke off, and the agent keeps what came and asks
        # the model to go on: with the kept text last, or with a message after it.
        if track.cut and any(
            message.role == "assistant" and message.text.strip() and message.text.strip() in track.cut
            for message in request.messages[-2:]
        ):
            return True, "continues a reply that broke off"
        if last.role != "user":
            return False, "no user turn last"
        # Results of the calls the model waits for continue the task. Results
        # it has already moved past continue it only without new text (a retry,
        # or a history the agent trimmed). With a new instruction, they are a
        # side request, such as a summary.
        results = {result.id for result in last.of(Result)}
        if track.pending and any(answers(result, track.pending) for result in last.of(Result)):
            return True, "results of the pending calls"
        if results & self.issued and not last.text.strip():
            return True, "earlier results, no new text"
        return False, "other"

    def aside(self, request: Request) -> Reply:
        """A reply to side requests (titles, summaries) that echoes the visible
        facts, after a tag of its own (`aside_tag`), so the suite sees where the
        reply goes. If the last message asks for a format with markdown headings,
        the reply fills in those headings, as a real model would."""
        self.asides += 1
        facts = " ".join([f"{AUX}_{self.asides}", *list(dict.fromkeys(FACT.findall(request.everything)))[:20]])
        asked = request.last.text if request.last else ""
        headings = list(dict.fromkeys(HEADING.findall(asked)))[:20]
        if not headings:
            return Reply(text=facts)
        return Reply(text="\n\n".join(f"{heading}\n- {facts}" for heading in headings))

    def follow(self, exchange: Exchange) -> Reply | Error:
        """Follow the script.

        A step with tool calls waits until a request carries all their results;
        a request without them (a retry) gets the step again, with new call IDs.
        Errors, faulty streams and text are served `times` times, then the
        script moves on. The last step, which must be text, repeats.
        """
        request = exchange.request
        track = self.track(exchange.launch)
        pending = track.pending
        if pending and all(
            request.result(call.id) or (token and any(marker(token) in result.text for result in request.results))
            for call, token in itertools.zip_longest(pending.calls, pending.tokens, fillvalue="")
        ):
            track.pending = None
            if pending.step == track.position:
                track.position += 1
        index = track.position
        step = track.steps[index]
        missing = next((server for server in self.wanted(step) if not self.reachable(request, server)), None)
        if missing is not None:
            if track.bootstrap[missing]:
                spec = track.bootstrap[missing].pop(0)
                call = Call(f"call_boot_{exchange.number}", spec["name"], json.dumps(spec["arguments"]))
                track.pending = Served(-1, exchange.number, [call], [], track.number)
                self.issued.add(call.id)
                return Reply(calls=(call,))
            if self.run.missing_at is None:
                self.run.missing_at = exchange.started
            return Reply(text="SUITE_MISSING_TOOL: the fixture tool was not advertised.")
        if step.error or not step.uses or step.faulty:
            track.repeats += 1
            if track.repeats >= step.times and index < len(track.steps) - 1:
                track.position, track.repeats = index + 1, 0
        if index == len(track.steps) - 1:
            track.finished = True
            self.run.finished.add(track.number)
        if step.error:
            return step.error
        return self.say(step, index, exchange, track)

    def say(self, step: Step, index: int, exchange: Exchange, track: Track) -> Reply:
        calls, tokens = [], []
        name = self.scenario.id.replace(".", "_").replace("-", "_")
        if track.number:
            name += f"_t{track.number}"
        for number, item in enumerate(step.uses):
            token = item.token or f"{name}_{index}_{number}"
            call_id = step.call_ids[number] if step.call_ids else f"call_{exchange.number}_{number}"
            calls.append(self.make_call(item, exchange.request, token, call_id))
            tokens.append(token)
        self.issued.update(call.id for call in calls)
        if step.faulty:
            track.cut = step.text
        if calls:
            track.pending = Served(index, exchange.number, calls, tokens, track.number)
            self.run.served.append(track.pending)
        return Reply(
            text=step.text,
            thinking=step.thinking if exchange.request.reasons else "",
            thinking_form=step.thinking_form,
            calls=tuple(calls),
            stop=step.stop,
            stream=step.stream,
        )

    # Tool calls, in the form the recipe's tools take.

    def wanted(self, step: Step) -> list[str]:
        """The MCP servers whose tools a step calls."""
        if self.source != "mcp":
            return []
        found = [item.server or self.servers[0] for item in step.uses if item.kind not in OWN_TOOLS and not item.name]
        return list(dict.fromkeys(found))

    def reachable(self, request: Request, server: str) -> bool:
        return self.fixture_tool(request, server) is not None or self.route(request, server) is not None

    def fixture_tool(self, request: Request, server: str) -> str | None:
        """The agent's name for a server's fixture tool, found by the server's
        name at the end of the tool's description. With one server, a tool
        whose description the agent dropped counts by its name."""
        mark = f"Server {server}."
        for tool in request.tools:
            if tool.description.startswith(DESCRIPTION[:40]) and mark in tool.description:
                return tool.name
        if len(self.servers) == 1:
            for tool in request.tools:
                if tool.description.startswith(DESCRIPTION[:40]) or tool.name.endswith("suite_fixture"):
                    return tool.name
        return None

    def route(self, request: Request, server: str) -> str | None:
        """Where to send a server's MCP calls when the agent hides MCP tools
        behind its own tool, as the recipe's `[mcp.route]` describes. The
        model finds the route in what the agent showed it, and remembers it
        after the agent compacts it away, as a real model would."""
        spec = self.recipe.route(server)
        if not spec or spec["tool"] not in {tool.name for tool in request.tools}:
            return None
        if "find" not in spec:
            return ""
        # The server may list other tools first: take the fixture's, as a
        # model that reads the list would.
        found = [match.group(1) for match in re.finditer(spec["find"], request.visible)]
        if found:
            self.routed[server] = next((name for name in found if name.endswith("fixture")), found[0])
        return self.routed.get(server)

    def make_call(self, item: Use, request: Request, token: str, call_id: str) -> Call:
        call = self.whole_call(item, request, token, call_id)
        if item.broken == "half":
            return replace(call, arguments=call.arguments[: len(call.arguments) // 2])
        if item.broken == "unclosed":
            return replace(call, arguments=call.arguments.rstrip()[:-1])
        if item.broken == "list":
            return replace(call, arguments=json.dumps(list(json.loads(call.arguments).values())))
        return call

    def whole_call(self, item: Use, request: Request, token: str, call_id: str) -> Call:
        values = {"kind": item.kind, "token": token}
        for name in ("size", "style", "value", "seconds"):
            if getattr(item, name) is not None:
                values[name] = getattr(item, name)
        if item.kind == "subagent":
            spec = self.recipe.subagent[item.stage]
            found = re.search(spec["find"], request.visible) if "find" in spec else None
            fields = {"prompt": item.content, "found": found.group(1) if found else ""}
            return Call(call_id, spec["name"], json.dumps(render(spec.get("arguments", {}), fields)))
        if item.kind in OWN_TOOLS:
            spec = self.recipe.native[item.kind]
            path = f"{self.paths['workspace']}/{item.value}"
            fields = {"path": path, "content": item.content, "old": item.old, "new": item.new}
            return Call(call_id, spec["name"], json.dumps(render(spec.get("arguments", {}), fields)))
        if item.name:
            return Call(call_id, item.name, json.dumps(values) if item.arguments is None else item.arguments)
        if self.source == "native":
            spec = self.recipe.native["shell"]
            fixture = f"{self.paths['python']} {self.paths['fixture']} run {self.paths['fixture_dir']}"
            if item.command:
                command = item.command
            elif item.heredoc:
                command = f"{fixture} - <<'SUITE_EOF'\n{json.dumps(values)}\nSUITE_EOF"
            else:
                command = f"{fixture} {shlex.quote(json.dumps(values))}"
            arguments = render(spec.get("arguments", {}), {"command": command})
            return Call(call_id, spec["name"], json.dumps(arguments))
        text = json.dumps(values) if item.arguments is None else item.arguments
        server = item.server or self.servers[0]
        name = self.fixture_tool(request, server)
        target = None if name else self.route(request, server)
        if target is None:
            return Call(call_id, name or "suite_fixture", text)
        try:
            parsed = json.loads(text)
        except ValueError:
            parsed = text
        spec = self.recipe.route(server)
        arguments = render(spec["arguments"], {"route": target, "arguments": parsed, "arguments_json": text})
        return Call(call_id, spec["tool"], json.dumps(arguments))


def answers(result: Result, served: Served) -> bool:
    """Whether a tool result answers a served call: by its ID, or, as for an
    agent that renamed a repeated ID, by the marker the call's run printed."""
    return any(result.id == call.id for call in served.calls) or any(
        marker(token) in result.text for token in served.tokens
    )
