"""Tool calls: delivery of results, side effects, arguments, many calls, and
tools that fail in the ways real commands do."""

import base64
import json
from dataclasses import replace

from ..fixture import marker
from ..messages import Call, Error, Reply, Result, Stream
from ..results import INFO, OPTIONAL, RECOMMENDED, REQUIRED, check, failed, gap, label, passed, skipped
from ..scenario import Scenario
from ..script import DONE, TASK, Use, fail_with, say, use


def delivered(run, step: int, kinds: list[str]):
    """Fail unless every call of a step returned its own marker, once each. A
    failed call's marker is the fixture's error message, so the model sees
    the error, with or without the API's error flag."""
    if not run.was_served(step):
        return skipped(f"The calls of step {step + 1} were never served.")
    results = run.step_result(step)
    tokens = run.tokens(step)
    if not results:
        return failed("no result", f"The result of step {step + 1} never reached the model.")
    for token, kind, result in zip(tokens, kinds, results, strict=True):
        if marker(token) not in result.text:
            return failed("wrong result", f"The result for {token} lacks its marker: {result.text[:120]!r}")
        if kind == "text" and f"SUITE_END_{token}" not in result.text:
            return failed("clipped", f"Only {len(result.text.encode())} bytes of the result for {token} arrived.")
        if len(run.invocations(token)) > 1:
            return failed("ran twice", f"The fixture ran {len(run.invocations(token))} times for one call.")
    return None


class RoundTrip(Scenario):
    id = "tools.round-trip"
    title = "A streamed tool call gets its result back"
    description = (
        "The model streams one fixture call in fragments. Its result must come back"
        " whole under the same call ID, and the fixture must run exactly once. The"
        " `text` result is 4 KB, which no agent should clip. In `blank-text`, the"
        " call follows text of only new lines, which Anthropic rejects as a text"
        " block if the agent sends it back."
    )
    tools = True
    variants = ("text", "error", "with-text", "blank-text")

    @property
    def provokes(self):
        return ("text",) if self.variant == "blank-text" else ()

    def script(self):
        if self.variant == "with-text":
            return [use("text", text="SUITE_NOTE: calling the fixture now."), DONE]
        if self.variant == "blank-text":
            return [use("text", text="\n\n"), DONE]
        if self.variant == "text":
            return [use(Use("text", size=4096)), DONE]
        return [use(self.variant), DONE]

    @check(REQUIRED)
    def returned(self, run):
        return delivered(run, 0, ["error" if self.variant == "error" else "text"]) or passed()


class Parallel(Scenario):
    id = "tools.parallel"
    title = "Several calls in one reply all get results"
    description = (
        "The model asks for two, three or twenty tool calls in one reply, or three"
        " whose first takes 3 s (`slow-first`): an agent that runs them at once gets"
        " their results out of order, over MCP as responses to requests it still"
        " waits on. Every call must run once and every result must come back under"
        " its own ID, as all three APIs require."
    )
    tools = True
    variants = ("two", "three-mixed", "twenty", "slow-first")
    kinds = {
        "two": ["text", "text"],
        "three-mixed": ["text", "error", "text"],
        "twenty": ["text"] * 20,
        "slow-first": ["slow", "text", "text"],
    }

    def script(self):
        calls = [Use("slow", seconds=3) if kind == "slow" else kind for kind in self.kinds[self.variant]]
        return [use(*calls), DONE]

    @check(REQUIRED)
    def returned(self, run):
        return delivered(run, 0, self.kinds[self.variant]) or passed()


class Sequence(Scenario):
    id = "tools.sequence"
    title = "Results stay matched through a long tool loop"
    description = (
        "The model calls the tool eight times in a row, with an error in the middle."
        " Each result must come back once, under its own call."
    )
    tools = True
    kinds = ["text", "text", "error", "text", "text", "text", "text", "text"]

    def script(self):
        return [*(use(kind) for kind in self.kinds), DONE]

    @check(REQUIRED)
    def matched(self, run):
        for step, kind in enumerate(self.kinds):
            if problem := delivered(run, step, [kind]):
                return replace(problem, detail=f"Step {step + 1} of 8: {problem.detail}")
        return passed()


ARGUMENTS = {
    "escaping": "quotes \" ' \\ / and\nnew line\ttab\rreturn",
    "unicode": "café é 👩‍💻 中文 العربية ‏RTL‎ \U0001f9ea",
    "json-text": '{"role":"assistant","content":[null,true,{"x":"a\\\\b"}]}',
    "long": "BEGIN_" + 'abcé\\"' * 3000 + "_END",
    "shell-like": "$(echo pwned) `id` ${HOME} && rm -rf / ; | > out",
}


class Arguments(Scenario):
    id = "tools.arguments"
    title = "Tool arguments arrive exactly as streamed"
    description = (
        "The model streams a tool call whose string argument has escapes, Unicode,"
        " JSON text, shell syntax or 18 KB of text. The fixture must receive the exact string."
    )
    tools = True
    sources = ("mcp",)
    variants = tuple(ARGUMENTS)

    def script(self):
        return [use(Use("echo", value=ARGUMENTS[self.variant])), DONE]

    @check(REQUIRED)
    def exact(self, run):
        if not run.served:
            return skipped("The call was never served.")
        calls = run.invocations(run.tokens(0)[0])
        if not calls:
            return failed("no call", "The fixture never ran.")
        value = calls[0]["arguments"].get("value")
        if value != ARGUMENTS[self.variant]:
            return failed(
                "changed",
                f"The fixture received {len(value or '')} chars that differ from the"
                f" {len(ARGUMENTS[self.variant])} streamed.",
            )
        if not run.step_result(0):
            return failed("no result", "The result never reached the model.")
        return passed()


class Incomplete(Scenario):
    id = "tools.incomplete"
    title = "A broken tool call never runs"
    description = (
        "The model's tool call is broken: the connection drops halfway through the"
        " arguments, the arguments are invalid JSON or not an object, the reply hits"
        " the output limit mid-call, or one stream event carries truncated JSON. In"
        " `beside-valid`, a call with invalid JSON comes next to a valid one in the"
        " same reply. The agent must not run the broken call, and its next request"
        " must still answer every call. Retrying or stopping are both fine. A valid"
        " call comes first, to show that the agent runs the tool."
    )
    tools = True
    variants = ("cut", "malformed-json", "not-object", "length", "corrupt-event", "beside-valid")

    def script(self):
        broken = Use("echo", value="SUITE_DRAFT_" + "x" * 160, token="draft")
        if self.variant == "cut":
            draft = use(broken, stream=Stream(cut_after=6))
        elif self.variant == "length":
            draft = use(replace(broken, broken="half"), stop="length")
        elif self.variant == "corrupt-event":
            draft = use(broken, stream=Stream(corrupt=4))
        elif self.variant == "beside-valid":
            draft = use("text", replace(broken, broken="unclosed"))
        else:
            draft = use(replace(broken, broken="unclosed" if self.variant == "malformed-json" else "list"))
        return [use("text"), draft, use("text"), DONE]

    @check(REQUIRED)
    def not_run(self, run):
        control = run.tokens(0)
        if not control or not run.invocations(control[0]):
            return skipped("The valid call before the broken one never ran.")
        if not run.was_served(1):
            return skipped("The broken call was never served.")
        ran = run.invocations("draft")
        if self.variant == "corrupt-event":
            # A complete call may still run with its arguments intact.
            ran = [c for c in ran if c["arguments"].get("value") != "SUITE_DRAFT_" + "x" * 160]
        if ran:
            return failed(
                "ran broken call",
                f"The fixture ran with arguments from the broken call: {json.dumps(ran[0]['arguments'])[:150]}",
            )
        return passed(detail="Stopped" if not run.was_served(2) else "Retried after the broken call")


class EffectOnce(Scenario):
    id = "tools.effect-once"
    title = "A finished tool call does not run again after a provider error"
    description = (
        "A tool call runs and its result goes back. Then the provider fails: the"
        " connection drops mid-reply, or it returns a 500 or a 429. A retry must"
        " resend the result, not run the tool again."
    )
    tools = True
    variants = ("disconnect", "server", "rate")

    def script(self):
        if self.variant == "disconnect":
            failure = say("SUITE_PARTIAL " * 20, stream=Stream(cut_after=4))
        else:
            failure = fail_with(self.variant, retry_after="1")
        return [use("effect"), failure, DONE]

    @check(REQUIRED)
    def once(self, run):
        # The provider error answers the first task request after the call.
        call = run.served[0] if run.served else None
        if not call or not any(e.kind == "task" and e.number > call.exchange for e in run.exchanges):
            return skipped("The provider error after the call was never served.")
        count = len(run.invocations(run.tokens(0)[0]))
        if count > 1:
            return failed("ran twice", f"The side effect ran {count} times for one call.")
        if count == 0:
            return failed("never ran", "The tool call never ran.")
        return passed(detail="Retried without rerunning" if run.finished_script else "Stopped after the error")


class Killed(Scenario):
    id = "tools.killed"
    title = "A tool killed by the system gives an error result"
    description = (
        "The tool's process is killed at once with SIGKILL, as the kernel's"
        " out-of-memory killer does. With the shell tool, that is the command;"
        " with MCP, it is the tool's server. The agent must live on, give the"
        " model a result for the call, and finish the task. The detail says"
        " whether the result has the API's error flag."
    )
    tools = True

    def script(self):
        return [use("killed"), DONE]

    @check(REQUIRED)
    def survives(self, run):
        if not run.served:
            return skipped("The call was never served.")
        if not run.invocations(run.tokens(0)[0]):
            return failed("no call", "The tool never ran.")
        results = run.step_result(0)
        if not results:
            if run.ended and run.last.termination == "exited":
                return failed("agent died", f"The agent exited without a result. {run.exit_detail()}".strip())
            return failed("no result", "The killed call never got a result.")
        if not run.finished_script:
            return failed("stopped", "The agent did not go on after the result.")
        flag = "with the error flag" if results[0].error else "without the error flag"
        return passed(detail=f"The result came {flag}: {results[0].text[:100]!r}")


class UnknownTool(Scenario):
    id = "tools.unknown-name"
    title = "A call to a tool that does not exist gets an error result"
    description = (
        "The model calls a tool that the agent never advertised. The agent should"
        " answer with an error result and let the model continue. In `long-name`,"
        " the name has 240 characters: sent back as it came, it breaks Anthropic's"
        " limit of 200 on every later request."
    )
    tools = True
    variants = ("default", "long-name")

    @property
    def provokes(self):
        return ("tool-name",) if self.variant == "long-name" else ()

    def script(self):
        name = "suite_" + "x" * 234 if self.variant == "long-name" else "suite_tool_that_does_not_exist"
        return [use(Use("text", name=name)), DONE]

    @check(RECOMMENDED)
    def answered(self, run):
        if not run.served:
            return skipped("The call was never served.")
        results = run.step_result(0)
        if results and run.finished_script:
            return passed("answered", f"The result: {results[0].text[:100]!r}")
        if results:
            return gap("result, then stop")
        return gap("no result", "The agent never answered the unknown call.")


class BadArguments(Scenario):
    id = "tools.bad-arguments"
    title = "What happens to a call whose arguments break the tool's schema"
    description = (
        "The model calls the fixture without its required `kind` argument"
        " (`missing-field`), or with arguments that are an empty string, as some"
        " models send for a call with no arguments (`empty`). The agent can reject"
        " the call, run it anyway, or stop."
    )
    tools = True
    sources = ("mcp",)
    variants = ("missing-field", "empty")

    def script(self):
        arguments = '{"token": "SUITE_NO_KIND"}' if self.variant == "missing-field" else ""
        return [use(Use("text", arguments=arguments)), DONE]

    @check(INFO)
    def handling(self, run):
        ran = any(not c["arguments"].get("kind") for c in run.fixture_calls)
        if not run.served:
            return skipped("The call was never served.")
        results = run.step_result(0)
        if not results:
            return label("stopped", "No result came back.")
        if ran:
            return label("ran it", "The fixture ran with the invalid arguments.")
        return label("rejected it", f"Error result before running: {results[0].text[:100]!r}")


class EmptyResult(Scenario):
    id = "tools.empty-result"
    title = "A tool result with nothing in it"
    description = (
        "The tool succeeds and returns nothing: a command that prints nothing, or"
        " an MCP tool with no content (`nothing`); or an MCP tool whose content is"
        " one empty text block (`empty-text`). The agent must still send the result"
        " in a request the API accepts: Anthropic's API, for one, rejects an empty"
        " text block. Then it should go on."
    )
    tools = True
    variants = ("nothing", "empty-text")
    provokes = ("text",)

    def fits(self):
        # Only an MCP tool returns content blocks.
        return super().fits() and not (self.variant == "empty-text" and self.source != "mcp")

    def script(self):
        return [use(Use("empty", value=self.variant)), DONE]

    @check(REQUIRED)
    def sent(self, run):
        if not run.served:
            return skipped("The call was never served.")
        if not run.step_result(0, anywhere=True):
            return failed("no result", f"The empty result never reached the model. {run.exit_detail()}".strip())
        if not run.finished_script:
            return failed("stopped", "The agent did not go on after the empty result.")
        shown = run.step_result(0, anywhere=True)[0].text
        return passed("sent", f"The model got {shown[:80]!r}." if shown else "The model got an empty result.")


class ReusedId(Scenario):
    id = "tools.reused-id"
    title = "The model reuses a call ID from an earlier turn"
    description = (
        "Some providers reuse call IDs like `call_0` on every turn. The model calls"
        " the tool twice with the same ID. The agent should still send a request the"
        " API accepts: it may give the second call its own ID, or drop it, but"
        " must not mix up the two results."
    )
    tools = True
    provokes = ("tool-id", "tool-pairs")

    def script(self):
        return [use("text", call_ids=("call_0",)), use("text", call_ids=("call_0",)), DONE]

    @check(RECOMMENDED)
    def kept_apart(self, run):
        second = next((s for s in run.served if s.step == 1), None)
        if second is None:
            return skipped(f"The model's second call never came. {run.exit_detail()}".strip())
        later = [e.request for e in run.tasks if e.number > second.exchange]
        first, latest = marker(run.tokens(0)[0]), marker(second.tokens[0])
        if any(latest in request.visible for request in later):
            return passed("kept apart")
        after = later[0] if later else None
        texts = (
            [part.text for message in after.messages for part in message.parts if isinstance(part, Result)]
            if after
            else []
        )
        if sum(first in text for text in texts) > 1:
            return gap("results mixed up", "The second call came back with the first call's result.")
        if run.invocations(second.tokens[0]):
            return gap("result lost", "The second call ran, but its result never reached the main loop.")
        # Not running the repeated call keeps the IDs unique, which is safe.
        return passed("dropped the repeat", f"The agent did not run the second call. {run.exit_detail()}".strip())


CALL_IDS = {
    # Gemini's and Vertex's compatible endpoints give long IDs.
    "long": ("call_" + "Ab9" * 40, "call_" + "Zy8" * 40),
    # Kimi names a call by its function and its position in the session.
    "punctuated": ("functions.suite_fixture:0", "functions.suite_fixture:1"),
}


class CallIds(Scenario):
    id = "tools.call-ids"
    title = "Results come back under call IDs of any shape"
    description = (
        "The model makes two parallel calls whose IDs are 125 characters long"
        " (`long`), as Gemini's compatible endpoint sends, or hold a dot and a"
        " colon (`punctuated`), as Kimi sends. Each result must come back paired"
        " with its call: under the call's exact ID, or under a new ID that the"
        " agent gives the call alike everywhere in the request, which a stateless"
        " API allows. Anthropic allows only letters, digits, `_` and `-`, so it has"
        " no punctuated IDs."
    )
    tools = True
    variants = tuple(CALL_IDS)

    def fits(self):
        # Anthropic call IDs hold only letters, digits, _ and -.
        return super().fits() and not (self.variant == "punctuated" and self.api == "anthropic")

    def script(self):
        return [use("text", "text", call_ids=CALL_IDS[self.variant]), DONE]

    @check(REQUIRED)
    def kept(self, run):
        if not run.was_served(0):
            return skipped("The calls were never served.")
        tokens, served = run.tokens(0), [c.id for c in run.served[0].calls]
        # A call is found by the token in its arguments, under its own ID or a
        # new one. The suite's model reads results under IDs it never gave as
        # a side request, so every request counts here.
        paired = None
        for request in (e.request for e in run.exchanges if e.request):
            calls = [renamed(request, token) for token in tokens]
            results = [request.result(call.id) if call else None for call in calls]
            if all(r and marker(t) in r.text for t, r in zip(tokens, results, strict=True)):
                paired = calls, results
                break
        if paired is None:
            return failed("no result", "No request holds both results, each with its call.")
        calls, results = paired
        for token, result in zip(tokens, results, strict=True):
            if f"SUITE_END_{token}" not in result.text:
                return failed("clipped", f"Only {len(result.text.encode())} bytes of the result for {token} arrived.")
            if len(run.invocations(token)) != 1:
                return failed("ran twice", f"The fixture ran {len(run.invocations(token))} times for one call.")
        ids = [c.id for c in calls]
        if ids == served:
            return passed()
        return passed("renamed", f"The agent sent the calls back as {', '.join(ids)}.")


def renamed(request, token: str):
    """The call in the request's history whose arguments hold the token."""
    return next((c for m in request.messages for c in m.of(Call) if token in c.arguments), None)


class OutputSize(Scenario):
    id = "tools.output-size"
    title = "How much of a large tool result reaches the model"
    description = (
        "The fixture returns 1 MiB, as short lines or as one dense line. The label"
        " shows how much reached the model, in steps of size, and which part"
        " survived; the detail has the exact size."
    )
    tools = True
    variants = ("lines", "dense")

    def script(self):
        return [use(Use("text", size=1_048_576, style=self.variant)), DONE]

    @check(INFO)
    def kept(self, run):
        results = run.step_result(0, anywhere=True)
        token = run.tokens(0)[0] if run.served else ""
        rejected = any(e.kind == "over-window" for e in run.exchanges)
        if not run.served:
            return skipped("The call was never served.")
        if not results:
            return label("over window, stopped" if rejected else "no result")
        if rejected and not run.finished_script:
            return label("sent whole, rejected", "The whole result exceeded the context window.")
        return label(*kept(results[0].text, token))


SIZES = ((16, "under 16 KB"), (64, "16-64 KB"), (256, "64-256 KB"), (1024, "256 KB-1 MB"))


def kept(text: str, token: str) -> tuple[str, str]:
    """How much of a long output reached the model, in steps of size, and
    which end; the exact size as a detail."""
    size = len(text.encode())
    step = next((name for limit, name in SIZES if size < limit * 1024), "1 MB+")
    head, tail = marker(token) in text, f"SUITE_END_{token}" in text
    where = "head+tail" if head and tail else "head" if head else "tail" if tail else "other"
    return f"{step}, {where}", f"The model got {size:,} bytes."


LONG_WAIT = 480.0  # Seconds: past a 300 s MCP call timeout, with room for startup under load.


class Unicode(Scenario):
    id = "tools.unicode"
    title = "Text the agent cuts or passes on stays valid Unicode"
    description = (
        "The fixture returns 1 MiB of emoji text, which the agent must cut"
        " (`cut`): a cut by UTF-16 units, as JavaScript strings count, can leave"
        " half of a surrogate pair. Or the MCP tool returns text that already holds"
        " half of one (`lone-surrogate`), as a JSON-RPC escape that a JavaScript or"
        " Python server that cut the text sends. No request may hold a lone"
        " surrogate: the APIs reject it as invalid JSON, and every later request"
        " of the session carries it again. A JSON parser that refuses the escape"
        " must still give the call a result; the suite waits"
        f" {LONG_WAIT:.0f} s, past the longest default MCP call timeout (Codex's 300 s)."
    )
    tools = True
    variants = ("cut", "lone-surrogate")
    provokes = ("text",)

    @property
    def timeout(self) -> float:
        return LONG_WAIT if self.variant == "lone-surrogate" else 60.0

    def settled(self, run):
        return bool(run.step_result(0, anywhere=True))

    def fits(self):
        # The text with a lone surrogate comes from the MCP tool.
        return super().fits() and not (self.variant == "lone-surrogate" and self.source != "mcp")

    def script(self):
        if self.variant == "cut":
            return [use(Use("text", size=1_048_576, style="unicode")), DONE]
        return [use("lone-surrogate"), DONE]

    @check(REQUIRED)
    def valid(self, run):
        if not run.served:
            return skipped("The call was never served.")
        if not run.step_result(0, anywhere=True):
            return failed("no result", "The result never reached the model.")
        return passed()


SUB_TASK = "SUITE_SUBTASK"  # In the task the model hands to a sub-agent.
SUB_ANSWER = "SUITE_SUB_ANSWER"  # Starts the sub-agent's answer.
SUB_TOKEN = "tools_subagent_sub"
SUB_FAILED = "SUITE_SUB_FAILED"  # In the error that fails every request of the sub-agent.


class Delegation(Scenario):
    """The model hands a task to the agent's own sub-agent tool, bound as
    `[native.subagent]`: one call, or one that starts the sub-agent and others
    that wait for it. `sub` answers the sub-agent's requests."""

    tools = True
    sources = ("native",)
    binds = ("subagent",)
    timeout = 120.0

    def __init__(self, *args, **options):
        super().__init__(*args, **options)
        self.sub_calls: list[str] = []

    def script(self):
        task = f"{SUB_TASK}: run the suite tool once, then answer with what it printed."
        stages = range(len(self.recipe.subagent))
        return [*(use(Use("subagent", content=task, stage=n)) for n in stages), DONE]

    def of_sub(self, request) -> bool:
        """A request of the sub-agent: it brings the task, or a result of its own call."""
        last = request.last
        if last is None:
            return False
        return SUB_TASK in last.text or any(r.id in self.sub_calls for r in last.of(Result))

    def task(self, exchange, model):
        return self.sub(exchange, model) if self.of_sub(exchange.request) else model.follow(exchange)

    def aside(self, exchange, model):
        if self.of_sub(exchange.request):
            return self.sub(exchange, model)
        last = exchange.request.last
        if last is not None and SUB_ANSWER in last.text:
            # The answer of a sub-agent that ran in the background, as a message: the main loop goes on.
            exchange.kind, exchange.notes["why"] = "task", "the sub-agent's answer"
            return model.follow(exchange)
        return model.aside(exchange.request)

    def sub(self, exchange, model):
        exchange.kind = "sub"
        request = exchange.request
        if done := [found for call in self.sub_calls if (found := request.result(call))]:
            return Reply(text=f"{SUB_ANSWER}: {done[-1].text.strip()[:300]}")
        call = model.make_call(Use("text"), request, SUB_TOKEN, f"call_sub_{exchange.number}")
        self.sub_calls.append(call.id)
        model.issued.add(call.id)
        return Reply(calls=(call,))

    def subs(self, run) -> list:
        return [e for e in run.exchanges if e.kind == "sub" and e.request]

    def handed(self, run):
        """The main loop's first call, which starts the sub-agent."""
        return next((c for s in run.served if s.step == 0 for c in s.calls), None)


class Subagent(Delegation):
    id = "tools.subagent"
    title = "A sub-agent's answer comes back to the main loop"
    description = (
        "The model hands a task to the agent's own sub-agent tool, bound as"
        " `[native.subagent]`: one call, or one that starts the sub-agent and"
        " others that wait for it. The sub-agent's requests, which carry the task,"
        " get a script of their own: run the fixture tool once, then answer. The"
        " tool must run once, and the answer must reach the main loop. A pass says"
        " whether the sub-agent's first request holds only its task or also the"
        " main conversation, and which model it asks."
    )

    @check(OPTIONAL)
    def answered(self, run):
        if not run.was_served(0):
            return skipped("The call to the sub-agent tool was never served.")
        if not self.subs(run):
            return failed("no sub-agent", "The sub-agent tool sent no request of its own.")
        ran = len(run.invocations(SUB_TOKEN))
        if ran != 1:
            return failed("tool never ran" if not ran else "tool ran twice", f"The sub-agent's tool ran {ran} times.")
        call = self.handed(run)
        result = run.result(call) if call else None
        if result is not None and SUB_ANSWER in result.text:
            how = "The answer came back as the tool's result."
        elif any(SUB_ANSWER in request.visible for request in run.requests):
            how = "The sub-agent ran in the background; its answer came back later."
        else:
            return failed("answer lost", "No request of the main loop holds the sub-agent's answer.")
        first = self.subs(run)[0].request
        start = "whole conversation" if any(TASK in text for text in first.user_texts) else "own task"
        model = "same model" if first.model == run.requests[0].model else f"model {first.model}"
        return passed(f"{start}, {model}", how)


class SubagentError(Delegation):
    id = "tools.subagent-error"
    title = "A sub-agent that fails does not stop the main loop"
    description = (
        "The model hands a task to the agent's own sub-agent tool, and the provider"
        " rejects every request of the sub-agent with an error that stays, as when the"
        " endpoint refuses the sub-agent's model or a parameter. The error's"
        " message holds a marker. The main loop should get the marker, from the"
        " call's result or, for a sub-agent that runs in the background, from a"
        " later message, and go on."
    )
    max_requests = 60

    def sub(self, exchange, model):
        exchange.kind = "sub"
        return Error("invalid", message=f"{SUB_FAILED}: the endpoint does not serve this model.")

    def told(self, run, call) -> list[str]:
        """What reached the main loop from the call's result on: tool results
        and messages, in the newest request that holds the result. A message
        that comes after the task is done is a side request, so this looks at
        every request."""
        holding = [e.request for e in run.exchanges if e.request and e.request.result(call.id)]
        if not holding:
            return []
        texts, after = [], False
        for message in holding[-1].messages:
            if message.role == "assistant":
                continue
            after = after or any(r.id == call.id for r in message.of(Result))
            if after:
                texts += [r.text for r in message.of(Result)] + ([message.text] if message.text else [])
        return texts

    @check(RECOMMENDED)
    def goes_on(self, run):
        if not self.subs(run):
            return skipped("The sub-agent sent no request.")
        call = self.handed(run)
        texts = self.told(run, call) if call else []
        if not texts:
            return gap("no result", "The main loop never got a result for the sub-agent call.")
        if not run.finished_script:
            return gap("stopped", f"The main loop stopped after: {texts[-1][:160]!r}")
        failing = next((t for t in texts if SUB_FAILED in t), None)
        if failing is None:
            return gap("failure hidden", f"The provider's error never reached the main loop: {texts[-1][:160]!r}")
        result = run.result(call)
        told = "told in the result" if result and SUB_FAILED in result.text else "told later"
        return passed(told, repr(failing[:160]))


class TerminalEscapes(Scenario):
    id = "tools.terminal-escapes"
    title = "Tool output cannot drive the user's terminal"
    description = (
        "In the TUI, the tool's result holds terminal control sequences, as a log"
        " or a file from an untrusted repository can: one sets the window title,"
        " one writes the clipboard (OSC 52), each with a marker. The TUI may show"
        " the output, but must not pass the sequences to the terminal: there they"
        " would replace what the user copies next. The suite reads every byte the"
        " TUI wrote to its terminal."
    )
    tools = True

    def script(self):
        return [use("escapes"), DONE]

    @check(RECOMMENDED)
    def contained(self, run):
        if not run.step_result(0):
            return skipped("The result never reached the model; tools.round-trip judges that.")
        token = run.tokens(0)[0]
        written = b"".join(launch.written for launch in run.launches)
        clip = base64.b64encode(f"SUITE_CLIP_{token}".encode())
        passed_on = [
            name
            for name, sequence in (
                ("clipboard", b"\x1b]52;c;" + clip),
                ("title", f"\x1b]0;SUITE_TITLE_{token}".encode()),
            )
            if sequence in written
        ]
        if passed_on:
            return gap("passed on", f"The TUI wrote the tool's {' and '.join(passed_on)} sequence to the terminal.")
        return passed()


REPEATS = 8  # Identical calls in a row, as a model stuck in a loop sends them.


class Repeat(Scenario):
    id = "tools.repeat"
    title = "What the agent does when the model repeats the same call"
    description = (
        f"The model sends the same call, with the same arguments, {REPEATS} times in"
        " a row, as a model stuck in a loop does, then ends the task. Some agents"
        " see the loop and stop running the call, or end the turn; others run each."
        " The label shows how many of the calls ran."
    )
    tools = True

    def script(self):
        same = Use("text", token="SUITE_REPEATED")
        return [*(use(same) for _ in range(REPEATS)), DONE]

    @check(INFO)
    def runs(self, run):
        if not run.served:
            return skipped("The model's call never went out.")
        ran = len(run.invocations("SUITE_REPEATED"))
        if ran >= REPEATS:
            return label("ran each", f"{ran} runs.")
        return label("stopped the loop", f"{ran} of the {REPEATS} calls ran.")
