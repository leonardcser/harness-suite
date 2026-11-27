"""Provider errors and limits: what the agent retries, how it waits, and how
it stops. The suite sees only the requests, their timing, and how the agent
exits."""

import json
from collections import Counter

from ..messages import Error, Stream
from ..results import RECOMMENDED, check, gap, passed, skipped
from ..scenario import Scenario
from ..script import DONE, PROMPT, Launch, Step, fail_with, say, use


def attempts(run) -> list:
    return [e for e in run.tasks if e.error or e.reply]


def resends(run) -> int:
    """The most times the agent sent one failing request unchanged: a request
    it changes to work around the error is not a blind retry."""
    sent = Counter(json.dumps(e.request.raw, sort_keys=True) for e in attempts(run) if e.error and e.request)
    return max(sent.values(), default=0)


def waits(run, tries: list) -> list[float]:
    return [run.between(a, b) for a, b in zip(tries, tries[1:], strict=False)]


BACKOFF_TRIES = 6  # Enough to see how the waits change.
NO_WAIT = 1.0  # Seconds: six tries with every wait shorter than this hammer the provider.


def ending(run) -> str:
    return "stopped" if run.ended else "still trying"


class Retry(Scenario):
    id = "errors.retry"
    title = "One transient provider error is retried"
    description = (
        "The first request fails once with a realistic error: a 429 with a"
        " `retry-after: 1` header, a 429 without any hint, a 500, an overloaded"
        " error (503, or 529 on Anthropic), a proxy's 502 page in HTML instead of"
        " the API's JSON (`gateway`), a connection reset before any response"
        " (`reset`), a connection closed before any event, or the API's error event"
        " in the middle of a stream. The agent should send the request again and finish."
    )
    variants = ("rate", "rate-no-hint", "server", "overloaded", "gateway", "reset", "connection", "stream-error")

    def script(self):
        if self.variant == "connection":
            first = say("", stream=Stream(cut_after=0))
        elif self.variant == "stream-error":
            first = say("SUITE_PARTIAL " * 10, stream=Stream(error_after=5, error=Error("overloaded")))
        elif self.variant == "rate":
            first = fail_with("rate", retry_after="1")
        elif self.variant == "rate-no-hint":
            first = fail_with("rate")
        else:
            first = fail_with(self.variant)
        return [first, DONE]

    @check(RECOMMENDED)
    def retried(self, run):
        if not run.finished_script:
            return gap("stopped")
        first, retry = run.requests[0], run.requests[1]
        if retry.messages != first.messages:
            return passed(detail="The retry changed the conversation.")
        return passed(detail=f"After {run.between(run.tasks[0], run.tasks[1]):.1f} s.")


LONG_WAIT = 30.0  # Seconds: a capped wait this long does not hammer a provider that asked for an hour.


class RetryAfter(Scenario):
    id = "errors.retry-after"
    title = "The agent waits as long as the provider asks"
    description = (
        "The provider answers 429 and asks for a wait of 15 s, as `retry-after`"
        " seconds, or as `retry-after-ms`, as Azure OpenAI and the OpenAI SDKs"
        " read it. The agent should not retry sooner."
        " Anthropic sends only `retry-after` in whole seconds, so it gets no other"
        " form. An agent's own first backoff is a few seconds; 15 s is well past"
        " it, so only the hint explains a wait that long. In `long`, it asks for an"
        " hour: the agent should wait at least"
        f" {LONG_WAIT:.0f} s or stop; telling the user is fine."
    )
    variants = ("seconds", "milliseconds", "long")
    per_harness = True
    timeout = 90.0

    def fits(self):
        return super().fits() and not (self.api == "anthropic" and self.variant == "milliseconds")

    def script(self):
        if self.variant == "milliseconds":
            first = Error("rate", retry_after_ms="15000")
        else:
            first = Error("rate", retry_after="3600" if self.variant == "long" else "15")
        return [Step(error=first), DONE]

    def settled(self, run):
        return self.variant != "long" and len(attempts(run)) >= 2

    @check(RECOMMENDED)
    def honored(self, run):
        tries = attempts(run)
        if len(tries) < 2:
            if self.variant == "long":
                return passed(ending(run))
            return skipped(f"The agent did not retry ({ending(run)}); errors.retry judges that.")
        waited = run.between(tries[0], tries[1])
        if self.variant == "long":
            if waited >= LONG_WAIT:
                return passed("waited", f"It waited {waited:.0f} s; the provider asked for 3600 s.")
            return gap("retried early", f"It retried after {waited:.0f} s; the provider asked for 3600 s.")
        if waited >= 14:
            return passed("waited", f"It waited {waited:.0f} s; the provider asked for 15 s.")
        return gap("retried early", f"It retried after {waited:.0f} s; the provider asked for 15 s.")


class Backoff(Scenario):
    id = "errors.backoff"
    title = "The agent backs off when errors do not stop"
    description = (
        "Every request fails, with a 429 that gives no hint when to retry, or with a"
        " 500. The agent should wait between attempts and not hammer the provider:"
        f" {BACKOFF_TRIES} tries with every wait under {NO_WAIT:.0f} s is a gap, while"
        " an agent that stops trying does not hammer. The detail lists the waits;"
        f" the suite watches for {BACKOFF_TRIES} tries, or at most 120 s."
    )
    variants = ("rate", "server")
    per_harness = True
    timeout = 120.0
    max_requests = 200

    def script(self):
        return [Step(error=Error(self.variant), times=10_000), DONE]

    def settled(self, run):
        return len(attempts(run)) >= BACKOFF_TRIES

    @check(RECOMMENDED)
    def backs_off(self, run):
        tries = attempts(run)
        gaps = waits(run, tries)
        detail = f"{len(tries)} {'try' if len(tries) == 1 else 'tries'}, {ending(run)}."
        if gaps:
            detail += " Waits: " + ", ".join(f"{w:.1f}" for w in gaps) + " s."
        if len(tries) >= BACKOFF_TRIES:
            return gap("no wait", detail) if max(gaps) < NO_WAIT else passed("waits", detail)
        # Fewer tries in the time the suite watches: the agent stopped, or waits long.
        return passed("stopped" if run.ended else "waits", detail)


class Final(Scenario):
    id = "errors.final"
    title = "The agent stops on an error that will not go away"
    description = (
        "The provider rejects every request: a 400 invalid request, a 401"
        " authentication error, a 404 for an unknown model, a 413 for a request"
        " that is too large, an empty account (OpenAI's 429"
        " `insufficient_quota`, Anthropic's low credit balance) from the start or"
        " after a first tool call, or the context error even after the agent"
        " compacts. Retrying cannot help, so the agent should not send the same"
        " request more than 3 times, or 6 after a context error, which compacting"
        " may fix. A request the agent changes, such as with fewer options, is not"
        " a blind retry. The suite watches for at most 90 s."
    )
    variants = ("invalid", "auth", "model", "too-large", "quota", "quota-later", "context")
    timeout = 90.0
    max_requests = 100

    @classmethod
    def uses_tools(cls, variant):
        return variant == "quota-later"

    def script(self):
        error = Step(error=Error(self.variant.removesuffix("-later")), times=10_000)
        if self.variant.endswith("-later"):
            return [use("text"), error, DONE]
        return [error, DONE]

    @property
    def allowed(self) -> int:
        return 6 if self.variant == "context" else 3

    def settled(self, run):
        # Two more than allowed: an agent past its last try gets time to exit.
        return resends(run) > self.allowed + 2

    @check(RECOMMENDED)
    def stops(self, run):
        tries, allowed = resends(run), self.allowed
        if not tries:
            return skipped("The agent sent no request.")
        sent = f"{len([e for e in attempts(run) if e.error])} failing requests, at most {tries} the same."
        # Where a slow agent is when the suite stops it is timing: every way
        # of not stopping in time is one gap.
        if run.last.termination == "settled" or not run.ended:
            return gap("kept retrying", f"The suite stopped it. {sent}")
        if tries > allowed:
            return gap(
                "kept retrying", f"It stopped after more than {allowed} tries. {sent} {run.exit_detail()}".strip()
            )
        summary = "stopped at once" if tries == 1 else f"stopped after {tries}"
        return passed(summary, f"{sent} {run.exit_detail()}".strip())


SECOND_PROMPT = "SUITE_SECOND_PROMPT"


class Recover(Scenario):
    id = "errors.recover"
    title = "The next prompt works after a failed turn"
    description = (
        "In the TUI, every request of the first prompt meets an empty account. Then"
        " the account works again and the user sends a second prompt. The agent"
        " should run it, in the same session."
    )
    needs = ("continue",)
    quiet = 20.0

    def launches(self):
        return [Launch(), Launch(prompt=f"{PROMPT}\n{SECOND_PROMPT}", resume=True, track=1)]

    def scripts(self):
        return [[Step(error=Error("quota"), times=10_000), DONE], [DONE]]

    @check(RECOMMENDED)
    def recovered(self, run):
        # A late retry of the first prompt also comes in the second launch.
        if not any(e.launch == 1 and any(SECOND_PROMPT in t for t in e.request.user_texts) for e in run.tasks):
            return gap("no request", "The second prompt sent no request.")
        return passed()
