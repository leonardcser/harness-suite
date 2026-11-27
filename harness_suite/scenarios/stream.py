"""Streaming: valid SSE variations, and streams that end badly."""

from ..fixture import marker
from ..messages import Stream
from ..results import INFO, RECOMMENDED, REQUIRED, check, demoted, failed, gap, label, passed, skipped
from ..scenario import Scenario
from ..script import DONE, TASK, Use, say, use

TEXT = "SUITE_TEXT: café é 👩‍💻 中文 \U0001f9ea end."
LONG_TEXT = "".join(f"\nSUITE_LINE_{n:05d} {'abcdefghij' * 5}" for n in range(5000))  # 300 KB.


class Framing(Scenario):
    id = "stream.framing"
    title = "Valid SSE variations are parsed"
    description = (
        "The provider streams text and a tool call with valid but less common SSE:"
        " comment lines, CRLF line ends, no space after `data:`, the bytes split"
        " one at a time across multi-byte"
        " characters, events that a newer API version adds (`new-events`: an"
        " unknown event type, or on Chat a delta with an unknown field), gzip, as a"
        " CDN sends it to a client that accepts it (`gzip`), or 300 KB of text in"
        " one event, over the 64 KB a default line reader takes (`long-line`). The"
        " tool must run, and the streamed text must come back unchanged in the next"
        " request."
    )
    tools = True
    STANDARD = ("comments", "crlf", "no-space", "byte-split", "new-events", "gzip", "long-line")
    variants = (*STANDARD, "azure-filter")

    @property
    def timeout(self):
        # A byte at a time takes up to 30 s on Responses, without load.
        return 300.0 if self.variant == "byte-split" else 60.0

    def fits(self):
        return super().fits() and (self.variant != "azure-filter" or self.api == "chat")

    def script(self):
        if self.variant == "long-line":
            return [use("text", text=TEXT + LONG_TEXT, stream=Stream(framing="long-line")), DONE]
        if self.variant == "byte-split":
            stream = Stream(chunk=1)
        elif self.variant == "new-events":
            stream = Stream(unknown=True)
        elif self.variant == "azure-filter":
            stream = Stream(azure_filter=True)
        else:
            stream = Stream(framing=self.variant)
        return [use("text", text=TEXT, stream=stream), DONE]

    @check(REQUIRED, variants=STANDARD)
    def parsed(self, run):
        if problem := understood(run, TEXT + LONG_TEXT if self.variant == "long-line" else TEXT):
            return problem
        if self.variant == "gzip" and not any(e.notes.get("gzip") for e in run.tasks):
            return passed("no gzip asked", "The agent does not accept gzip, so the stream was plain.")
        return passed()

    @check(
        RECOMMENDED,
        id="stream.azure-filter",
        title="Azure's first chunk with no choices is parsed",
        description=(
            "On Chat, the stream starts with the chunk Azure OpenAI sends for its"
            " content filter: no choices, only `prompt_filter_results`. The tool must"
            " run, and the text must come back unchanged. One provider sends it, so"
            " it is recommended."
        ),
        variants=("azure-filter",),
    )
    def azure(self, run):
        if problem := understood(run):
            return demoted(problem)
        return passed()


def understood(run, text: str = TEXT):
    """A fail unless the reply's tool call ran and its text came back whole;
    a skip if the reply was never served."""
    if not run.served:
        return skipped("The reply with the call was never served.")
    results = run.step_result(0)
    if not results:
        return failed("no result", "The tool call was not parsed or its result never came back.")
    if marker(run.tokens(0)[0]) not in results[0].text:
        return failed("wrong result")
    follow = next((r for r in run.requests if r.result(run.calls[-1].id)), None)
    said = [m.text for m in follow.messages if m.role == "assistant"] if follow else []
    if not any(text in sent for sent in said):
        return failed(
            "text changed",
            f"The text came back as {said[-1][:80]!r}" if said else "The text is missing from the history.",
        )
    return None


# The arguments of three parallel calls: long enough to stream in many
# fragments, and each unlike the others.
VALUES = (
    'SUITE_FIRST: alpha {"x": [1, 2]} ' + "a" * 40,
    'SUITE_SECOND: 中文 \\ "beta" ' + "b" * 60,
    "SUITE_THIRD: " + "γ" * 80,
)


def echoed_without(run, token: str) -> str | None:
    """The arguments of the scripted call with `token`, as the agent sent the
    call back in its history, if they lost the token."""
    served = next((c for s in run.served if s.step == 0 for c in s.calls if token in c.arguments), None)
    if served is None:
        return None
    for request in run.requests:
        for call in request.calls:
            if call.id == served.id and token not in call.arguments:
                return call.arguments or "nothing"
    return None


def assembled(run):
    """A fail unless each parallel call ran once with exactly its own
    arguments; a skip if the calls were never served."""
    if not run.served:
        return skipped("The reply with the calls was never served.")
    for number, (token, value) in enumerate(zip(run.tokens(0), VALUES, strict=True), 1):
        calls = run.invocations(token)
        if not calls:
            empty = sum(not c["arguments"] for c in run.fixture_calls)
            if empty:
                return failed("arguments lost", f"The fixture ran {empty} times with no arguments at all.")
            if echoed := echoed_without(run, token):
                return failed("arguments lost", f"The agent sent the call back with {echoed[:80]!r}, without them.")
            return failed("call lost", f"The fixture never ran for call {number} of 3.")
        got = calls[0]["arguments"].get("value")
        if got != value:
            return failed("mixed up", f"Call {number} of 3 ran with {str(got)[:60]!r}, not its own arguments.")
    if len(run.step_result(0)) < len(VALUES):
        return failed("no result", "Not every result came back to the model.")
    return None


class Interleaved(Scenario):
    id = "stream.interleaved-calls"
    title = "Parallel calls whose fragments arrive in turn"
    description = (
        "The model streams three parallel calls, and their argument fragments"
        " alternate: on Chat by `index`, on Responses by `output_index` and"
        " `item_id`, as OpenAI and compatible servers send them. Each call must run"
        " once with exactly its own arguments. Anthropic streams one block at a"
        " time, so it has no such stream."
    )
    tools = True
    sources = ("mcp",)
    apis = ("chat", "responses")

    def script(self):
        return [use(*(Use("echo", value=v) for v in VALUES), stream=Stream(calls="interleaved")), DONE]

    @check(REQUIRED)
    def apart(self, run):
        return assembled(run) or passed()


class CallChunks(Scenario):
    id = "stream.call-chunks"
    title = "Tool call chunks in the shapes compatible servers send"
    description = (
        "The model streams three parallel calls in a shape that the API allows but"
        " its own servers do not send. `whole`: the arguments arrive with no"
        " fragments, on Chat in the call's first chunk, on Responses only in"
        " `function_call_arguments.done` and `output_item.done`. On Chat only, the ID and name come again in"
        " every chunk (`repeated-identity`, as vLLM and Ollama send), or as empty"
        " strings after the first (`blank-identity`); or the calls end with"
        ' `finish_reason: "stop"`, not `tool_calls`, as vLLM, Ollama and Gemini\'s'
        " compatible endpoint send (`finish-stop`). Each call must run once with"
        " exactly its own arguments."
    )
    tools = True
    sources = ("mcp",)
    variants = ("whole", "repeated-identity", "blank-identity", "finish-stop")
    apis = ("chat", "responses")

    def fits(self):
        # Only Chat sends a call's ID and name in its chunks, and a finish reason.
        return super().fits() and (self.variant == "whole" or self.api == "chat")

    def script(self):
        echoes = (Use("echo", value=v) for v in VALUES)
        if self.variant == "finish-stop":
            return [use(*echoes, stop="end"), DONE]
        return [use(*echoes, stream=Stream(calls=self.variant)), DONE]

    @check(RECOMMENDED)
    def read(self, run):
        if problem := assembled(run):
            return demoted(problem)
        return passed()


# About 50 events, 1.5 s apart: over a minute in all, longer than the total
# timeout some HTTP clients put on a whole request, while no gap is long. The
# words never repeat: an agent may take a repeated phrase for a model stuck in
# a loop, which no real reply streams.
SLOW_TEXT = "SUITE_SLOW: " + " ".join(f"word{n:03d}" for n in range(30))
SLOW_PACE = 1.5


class Slow(Scenario):
    id = "stream.slow"
    title = "A slow but steady stream is read to its end"
    description = (
        f"The model streams its text and a tool call one event every {SLOW_PACE:g} s,"
        " for over a minute, as a large model or a long reasoning reply does. No"
        " gap is long, so the agent should wait for the end: a timeout on the"
        " whole request, not on silence, cuts it off. The proof is the next"
        " request: it holds the whole text and the tool's result."
    )
    tools = True
    timeout = 180.0

    def script(self):
        return [use("text", text=SLOW_TEXT, stream=Stream(pace=SLOW_PACE)), DONE]

    @check(RECOMMENDED)
    def read(self, run):
        problem = understood(run, SLOW_TEXT)
        if problem and problem.summary == "no result":
            ended = " ".join(run.ending(i) for i in range(len(run.launches)))
            return gap("not read to the end", f"The call in the slow stream never ran. {ended}")
        return demoted(problem) if problem else passed()


# How each variant ends the reply early: its text, stream and stop reason,
# and the API that has it.
EARLY = "SUITE_EARLY: the reply ends here"
ENDINGS = {
    "cut": (EARLY, Stream(drop_terminal=True), None),
    "length": (EARLY, None, "length"),
    "error": (EARLY, None, "error"),
    "window": (EARLY, None, "window"),
    "empty": ("", None, None),
}
ONLY = {"error": "chat", "window": "anthropic"}


def asked_again(run, first) -> bool:
    """Whether the agent asked the model again after the exchange `first`, in
    the same launch: any later request that offers tools and carries the
    task's prompt, a retry or a request to go on, however it words it."""
    return any(
        e.number > first.number
        and e.launch == first.launch
        and e.kind == "task"
        and e.request is not None
        and e.request.tools
        and any(TASK in text for text in e.request.user_texts)
        for e in run.exchanges
    )


class EndedEarly(Scenario):
    id = "stream.ended-early"
    title = "A reply that ends early is not taken as a finished turn"
    description = (
        "The model's text reply ends before it is complete: the stream closes"
        " without its finish events (`cut`), stops at the output limit (`length`:"
        " `length`, `max_tokens`, `max_output_tokens`), ends with Chat's `error`"
        " finish reason, as OpenRouter ends a stream that failed (`error`), or"
        " with Anthropic's `model_context_window_exceeded` (`window`). Headless,"
        " the agent should ask the model again, or exit with a non-zero status, so"
        " a script can tell. Exit status 0 with no further request reports a"
        " broken answer as a success. Any later request of the launch that offers"
        " tools and carries the task's prompt counts as asking again."
    )
    variants = tuple(ENDINGS)
    CUT_SHORT = tuple(v for v in ENDINGS if v != "empty")

    def fits(self):
        return super().fits() and ONLY.get(self.variant, self.api) == self.api

    def script(self):
        text, stream, stop = ENDINGS[self.variant]
        return [say(text, stop=stop, stream=stream or Stream()), DONE]

    def ending(self, run):
        """What the agent did after the reply: "asked again", "exit status",
        "no exit" or "exit 0", with a detail; None if it was never served."""
        first = next((e for e in run.tasks if e.reply), None)
        if first is None:
            return None
        if asked_again(run, first):
            return "asked again", ""
        if not run.ended:
            return "no exit", f"The agent still ran {self.timeout:.0f} s after the reply."
        if run.last.exit != 0:
            return "exit status", run.exit_detail()
        return "exit 0", "The agent exited with status 0 and did not ask the model again."

    @check(RECOMMENDED, variants=CUT_SHORT)
    def told(self, run):
        if (found := self.ending(run)) is None:
            return skipped("The early end was never served.")
        summary, detail = found
        if summary in ("asked again", "exit status"):
            return passed(summary, detail)
        return gap(summary, detail)

    @check(
        INFO,
        id="stream.empty-reply",
        title="What the agent does with a reply that holds nothing",
        description=(
            "The model's reply is complete, with a normal stop reason, but holds no"
            " text and no call, as models sometimes send. Some agents ask the model"
            " again; others end the turn with an empty answer. Headless, the label"
            " shows which, and with what exit status."
        ),
        variants=("empty",),
    )
    def empty(self, run):
        if (found := self.ending(run)) is None:
            return skipped("The empty reply was never served.")
        return label(*found)
