"""Prompt caching: whether each request extends the one before, so a provider
can serve its prefix from cache.

Providers cache a prompt by its exact prefix: the tools, the system prompt,
then the messages. A request that changes something near the start pays full
price and full latency for everything after it.
"""

from ..apis.anthropic import cache_marks
from ..results import RECOMMENDED, check, gap, passed, skipped, unsupported
from ..scenario import Scenario
from ..script import DONE, PROMPT, Launch, say, use

ONE, TWO = "SUITE_CACHE_ONE", "SUITE_CACHE_TWO"


def canonical(value):
    """A request part as a prompt cache sees it: without `cache_control`, whose
    move keeps the cache, and with text content as text blocks, since a string
    and one text block render the same."""
    if isinstance(value, dict):
        found = {k: canonical(v) for k, v in value.items() if k != "cache_control"}
        if isinstance(found.get("content"), str):
            found["content"] = [{"type": "text", "text": found["content"]}]
        return found
    if isinstance(value, list):
        return [canonical(v) for v in value]
    return value


def parts(request) -> tuple:
    """(tools, system prompt, messages) of a request, as a cache sees them: on
    Responses, with the history the provider keeps. A leading system or
    developer message counts as the system prompt."""
    raw = canonical(request.raw)
    system = raw.get("system", raw.get("instructions"))
    if isinstance(system, str):
        system = [{"type": "text", "text": system}]
    items = canonical(request.items) if request.items is not None else raw.get("messages") or []
    if system is None and items and isinstance(items[0], dict) and items[0].get("role") in ("system", "developer"):
        system, items = items[0].get("content"), items[1:]
    return raw.get("tools"), system, items


def prefix_change(before, after) -> str | None:
    """Where `after` stops extending `before`, as a prompt cache sees it."""
    (tools, system, old), (tools2, system2, new) = parts(before), parts(after)
    if tools != tools2:
        return "the tools"
    if system != system2:
        return "the system prompt"
    for index, item in enumerate(old):
        if index >= len(new) or new[index] != item:
            return f"message {index + 1}"
    return None


def stable(exchanges) -> tuple[bool, str]:
    for before, after in zip(exchanges, exchanges[1:], strict=False):
        if where := prefix_change(before.request, after.request):
            return False, f"Request {after.number} changes {where} of request {before.number}."
    return True, f"{len(exchanges)} requests, each extends the one before."


def loop(run) -> list:
    """The first launch's task requests from the one that got the first
    fixture call: before it, an agent may still be loading the tool, as Claude
    Code does with ToolSearch."""
    first = run.served[0].exchange if run.served else None
    return [e for e in run.tasks if e.launch == 0 and first is not None and e.number >= first]


class Session(Scenario):
    """A tool loop of three calls, then a continued turn with one more call."""

    id = "cache.tool-loop"
    title = "Each request in a tool loop extends the one before"
    description = (
        "The model calls the tool three times in a row. Each request should repeat"
        " the one before it and only add to its end: the same tools, the same"
        " system prompt, the same messages. Moving a `cache_control` mark does not"
        " count, nor does the step where an agent loads the deferred fixture tool."
    )
    tools = True

    def script(self):
        step = use("text")
        return [step, step, step, say("SUITE_FIRST_DONE"), step, DONE]

    def launches(self):
        first = Launch(prompt=f"{PROMPT}\n{ONE}")
        # Without a way to continue the session, only the tool loop is measured.
        if self.recipe.supports("continue") is not True:
            return [first]
        return [first, Launch(prompt=f"{PROMPT}\n{TWO}", resume=True)]

    @check(RECOMMENDED)
    def tool_loop(self, run):
        tasks = loop(run)
        if len(tasks) < 2:
            return skipped("The tool loop never ran: fewer than two requests with tools.")
        ok, detail = stable(tasks)
        return passed("stable", detail) if ok else gap("prefix changed", detail)

    @check(
        RECOMMENDED,
        id="cache.turns",
        title="A new turn extends the conversation before it",
        description=(
            "In the same session, the user sends a second prompt. Its first request"
            " should extend the last request of the first turn, so the whole earlier"
            " conversation stays cached. Headless, the second turn is a new process"
            " that resumes the session, so this also covers a restart."
        ),
    )
    def turns(self, run):
        before = [e for e in run.tasks if e.launch == 0]
        after = next((e for e in run.tasks if e.launch == 1), None)
        if len(run.launches) < 2:
            return unsupported(detail="The recipe cannot continue a headless session.")
        if not before or after is None:
            return skipped("The session did not reach a second turn.")
        if where := prefix_change(before[-1].request, after.request):
            return gap("prefix changed", f"Request {after.number} changes {where} of request {before[-1].number}.")
        return passed("stable")

    @check(
        RECOMMENDED,
        id="cache.key",
        title="A prompt cache key, if sent, stays the same through a session",
        description=(
            "OpenAI routes requests with the same `prompt_cache_key` to the same"
            " cache, which keeps more of a long session cached. A key is optional; one"
            " that changes from request to request, or after a restart, sends the"
            " session's requests to other caches. Chat and Responses only."
        ),
        apis=("chat", "responses"),
    )
    def key(self, run):
        by_launch: dict[int, list] = {}
        for e in run.tasks:
            by_launch.setdefault(e.launch, []).append(e.request.raw.get("prompt_cache_key"))
        keys = [key for found in by_launch.values() for key in found]
        if not any(keys):
            return passed("none")
        if any(len(set(found)) > 1 for found in by_launch.values()):
            return gap("changes", f"{len(set(keys))} keys in {len(keys)} requests.")
        if len(set(keys)) > 1:
            return gap("changes after a restart", f"One key per launch: {', '.join(map(repr, set(keys)))}.")
        return passed("one per session", f"{len(keys)} requests with {keys[0]!r}.")

    @check(
        RECOMMENDED,
        id="cache.marked",
        apis=("anthropic",),
        title="The request asks the provider to cache the prompt",
        description=(
            "Anthropic caches a prompt only up to a `cache_control` mark; OpenAI"
            " caches without one, so only Anthropic is measured. After a tool call,"
            " the request that carries the history should mark it. The detail shows"
            " where the marks are."
        ),
    )
    def marked(self, run):
        tasks = loop(run)
        if len(tasks) < 2:
            return skipped("The tool loop never reached a second request.")
        places = [name for name, count in cache_marks(tasks[1].request.raw) if count]
        return passed("marked", "cache_control on " + ", ".join(places)) if places else gap("no marks")
