"""Reasoning: how the model's reasoning goes back to it, and the settings that change it."""

from ..messages import Error, Result, Stream, Thinking, opaque, signature
from ..results import RECOMMENDED, REQUIRED, check, demoted, failed, passed, skipped
from ..scenario import Scenario
from ..script import DONE, use

THOUGHT = "SUITE_THINKING: the scripted plan is to call the fixture once."
AGAIN = "SUITE_THINKING_AGAIN: the plan for the second call."


def places(request, text: str) -> list[str]:
    """Where the reasoning `text` is in a request: "native" reasoning, in the
    API's own form (found by its text or by the identity the provider gave
    it, a signature or opaque data), assistant "text", a "user" message or
    tool result, or the "system" prompt."""
    identity = {signature(text), opaque(text)}
    found = []
    if text in request.system:
        found.append("system")
    for message in request.messages:
        if message.role == "assistant":
            if any(text in t.text or t.signature in identity for t in message.of(Thinking)):
                found.append("native")
            if text in message.text:
                found.append("text")
        elif text in message.text or any(text in r.text for r in message.of(Result)):
            found.append("user")
    return list(dict.fromkeys(found))


UNREASONED = "The agent did not turn reasoning on, so the API sent none."


def reasoned(run, launch: int = 0) -> bool:
    """Whether a reply of `launch` carried reasoning: the API sends it only
    when the request asks for it (`Request.reasons`)."""
    return any(e.reply and e.reply.thinking for e in run.tasks if e.launch == launch)


def where(request, text: str) -> str:
    """A label for where the reasoning went, best place first."""
    found = places(request, text)
    if not found:
        return "dropped"
    identity = {signature(text), opaque(text)}
    signed = any(t.signature in identity for m in request.messages for t in m.of(Thinking))
    names = {
        "native": "kept" if request.api == "chat" else ("kept, signed" if signed else "kept, unsigned"),
        "text": "as visible text",
        "user": "in a user message",
        "system": "in the system prompt",
    }
    return " + ".join(names[place] for place in found)


CUT = "SUITE_THINKING_CUT: " + "the plan goes on, " * 24  # Long enough to cut mid-reasoning.
CUT_AT = 8  # Events: past the start of the reasoning, well before its end.


class ToolLoop(Scenario):
    id = "reasoning.tool-loop"
    title = "Reasoning goes back unchanged while the tools run"
    description = (
        "The model reasons, then calls the tool. While the tool loop runs, the next"
        " request must carry that reasoning as it came: Anthropic thinking and"
        " `redacted_thinking` blocks in order with their signatures, Responses reasoning"
        " items with their encrypted content, and Chat `reasoning_content` (DeepSeek,"
        " Kimi, GLM and MiniMax require it). The real APIs reject a tool loop that"
        " drops or changes it. Variants: reasoning shown as text, `redacted` (opaque"
        " data), `omitted` (an Anthropic signature with empty text), two reasoning"
        " steps in one turn. A request the API rejects fails the check."
    )
    tools = True
    variants = ("shown", "redacted", "omitted", "interleaved", "late-encrypted")
    provokes = ("reasoning",)  # Each variant tests a replay rule of the API.
    # The API that has each variant's shape, if only one has it.
    only = {"late-encrypted": "responses", "omitted": "anthropic"}

    def fits(self):
        if self.variant == "redacted" and self.api == "chat":
            return False  # Chat Completions has no hidden reasoning.
        return super().fits() and self.only.get(self.variant, self.api) == self.api

    def script(self):
        if self.variant == "interleaved":
            return [use("text", thinking=THOUGHT), use("text", thinking=AGAIN), DONE]
        if self.variant == "late-encrypted":
            stream = Stream(late_reasoning=True)
            return [use("text", thinking=THOUGHT, text="SUITE_NOTE: calling the fixture.", stream=stream), DONE]
        return [use("text", thinking=THOUGHT, thinking_form=self.variant), DONE]

    @check(REQUIRED, variants=("shown", "redacted", "omitted", "interleaved"))
    def unchanged(self, run):
        if not run.served:
            return skipped("The call was never served.")
        if not reasoned(run):
            return skipped(UNREASONED)
        last = run.calls[-1]
        follow = next((r for r in run.requests if last and r.result(last.id)), None)
        if follow is None:
            return failed("no result", "The tool loop did not go on after the reasoning.")
        if self.variant in ("redacted", "omitted"):
            return passed()
        return passed(detail="; ".join(f"{name}: {where(follow, text)}" for name, text in self.thoughts()))

    @check(
        RECOMMENDED,
        id="reasoning.late-encrypted",
        title="Reasoning whose encrypted content comes only in the final response",
        description=(
            "Some Azure OpenAI deployments put a reasoning item's encrypted content"
            " only in `response.completed`, not in `response.output_item.done`"
            " (pi #4272). With `store: false`, the next request of the tool loop must"
            " carry that content: a reference by ID alone is rejected. This is a"
            " quirk of some deployments, not the API's contract, so it is a gap."
        ),
        variants=("late-encrypted",),
    )
    def late_encrypted(self, run):
        first = next(iter(run.tasks), None)
        stateless = first and first.request.raw.get("store") is False
        sealed = first and "reasoning.encrypted_content" in (first.request.raw.get("include") or [])
        if not (stateless and sealed):
            return skipped(
                "The late content matters only with `store: false` and encrypted reasoning,"
                " which the agent does not ask for."
            )
        return demoted(self.unchanged(run))

    def thoughts(self) -> list[tuple[str, str]]:
        if self.variant == "interleaved":
            return [("first step", THOUGHT), ("second step", AGAIN)]
        return [("reasoning", THOUGHT)]


class Cut(Scenario):
    id = "reasoning.cut"
    title = "A reply cut in its reasoning leaves a history the API accepts"
    description = (
        "The reply streams reasoning and breaks off before the reasoning ends,"
        " with the API's overloaded error, so the agent holds part of it, with no"
        " Anthropic signature and no finished Responses item. The agent retries,"
        " and the turn must finish; a history the API rejects, such as a thinking"
        " block without its signature or a half reasoning item, fails the check."
        " The detail shows where the partial reasoning went."
    )
    tools = True
    provokes = ("reasoning",)  # The partial reasoning tests a replay rule of the API.

    def script(self):
        broken = Stream(error_after=CUT_AT, error=Error("overloaded"))
        return [use("text", thinking=CUT, stream=broken), use("text", thinking=THOUGHT), DONE]

    @check(REQUIRED)
    def accepted(self, run):
        if not reasoned(run):
            return skipped(UNREASONED)
        cut = next((e for e in run.tasks if e.reply and e.reply.thinking == CUT), None)
        after = [e for e in run.tasks if cut and e.number > cut.number]
        if not after:
            return skipped("No request came after the cut; errors.retry judges retries.")
        if not run.finished_script:
            return failed("stopped", f"The turn after the cut did not finish. {run.exit_detail()}".strip())
        return passed(detail=f"The partial reasoning: {where(after[0].request, 'SUITE_THINKING_CUT')}.")
