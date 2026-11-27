"""Sessions across launches: continue, switch model, effort or API, crash."""

from ..recipe import MODELS
from ..results import INFO, RECOMMENDED, REQUIRED, check, failed, gap, label, passed, skipped
from ..scenario import Scenario
from ..script import DONE, PROMPT, Launch, Use, say, slow_started, use
from .reasoning import UNREASONED, reasoned, where

ONE, TWO = "SUITE_TURN_ONE", "SUITE_TURN_TWO"
FIRST_DONE, SECOND_DONE = "SUITE_FIRST_DONE", "SUITE_SECOND_DONE"
THOUGHT = "SUITE_THINKING: first turn plan."
PAIRS = {
    "chat-to-anthropic": ("chat", "anthropic"),
    "anthropic-to-responses": ("anthropic", "responses"),
    "responses-to-chat": ("responses", "chat"),
}
HISTORY = (
    " The label shows where the first turn's reasoning is in the new request."
    ' Anthropic and OpenAI recommend keeping it in place ("kept, signed"), also'
    " for another model, and DeepSeek with tools requires it; dropping it is"
    " allowed. Anthropic calls reasoning in visible text harmful, and no provider"
    " documents reasoning in a user message or the system prompt."
)


def first_request(run, launch: int):
    """The first request of the main loop that `launch` sent, also one the
    API rejected: `request.valid` judges that."""
    sent = (e for e in run.exchanges if e.launch == launch and e.request and e.kind in ("task", "rejected"))
    return next((e.request for e in sent), None)


def second_request(run):
    return first_request(run, 1)


def history_gaps(run, request) -> list[str]:
    """What the second launch's first request lacks from the first launch."""
    gaps = []
    if not any(ONE in text for text in request.user_texts):
        gaps.append("first prompt")
    # An agent may rewrite call IDs between APIs or models, so match the call by content.
    call = run.calls[0] if run.calls else None
    sent = next((c for c in request.calls if call and (c.name, c.args()) == (call.name, call.args())), None)
    if call and sent is None:
        gaps.append("tool call")
    result = request.result(sent.id) if sent else None
    if call and (result is None or "SUITE_RESULT_" not in result.text):
        gaps.append("tool result")
    if not any(FIRST_DONE in m.text for m in request.messages if m.role == "assistant"):
        gaps.append("final answer")
    return gaps


def continued(run, model: str | None = None):
    """Pass if the second launch sent the whole first turn to the right model;
    `request.valid` judges whether the API accepts it."""
    ended = any(e.launch == 0 and e.reply and FIRST_DONE in e.reply.text for e in run.exchanges)
    if not ended:
        return skipped("The first turn never finished, so there was no session to continue.")
    request = second_request(run)
    if request is None:
        return failed("no second turn", "The continued launch sent no task request.")
    if model and request.model != model:
        return failed("old model", f"The continued request used {request.model}.")
    if gaps := history_gaps(run, request):
        return failed("lost " + ", ".join(gaps))
    return passed()


def reasoning_place(run):
    if not reasoned(run):
        return skipped(UNREASONED)
    request = second_request(run)
    if request is None:
        return skipped("No second turn; session.continue judges that.")
    return label(where(request, THOUGHT))


class TwoTurns(Scenario):
    """A turn with reasoning and a tool call, then a continued second turn."""

    tools = True
    needs = ("continue",)

    def script(self):
        return [use("text", thinking=THOUGHT), say(FIRST_DONE), say(SECOND_DONE)]

    def launches(self):
        return [
            Launch(prompt=f"{PROMPT}\n{ONE}", **self.first()),
            Launch(prompt=f"{PROMPT}\n{TWO}", resume=True, **self.second()),
        ]

    def first(self) -> dict:
        return {}

    def second(self) -> dict:
        return {}


class Continue(TwoTurns):
    id = "session.continue"
    title = "A continued session sends the earlier turn"
    description = (
        "The agent runs a turn with reasoning and a tool call, then continues the"
        " session with a new prompt, with the same model or another one. The new"
        " request must hold the first prompt, the tool call, its result and the"
        " first answer, and use the new model."
    )
    variants = ("same-model", "other-model")

    def second(self):
        return {"model": MODELS[1]} if self.variant == "other-model" else {}

    @check(REQUIRED)
    def complete(self, run):
        return continued(run, MODELS[1] if self.variant == "other-model" else None)

    @check(
        INFO,
        id="reasoning.history",
        title="Where earlier reasoning goes in a later turn",
        description="In the same continued session, with the same model or another one." + HISTORY,
    )
    def reasoning(self, run):
        return reasoning_place(run)


class SwitchEffort(TwoTurns):
    id = "session.switch-effort"
    title = "Continuing with another effort level keeps the history"
    description = (
        "Like `session.continue`, but the first turn runs at high reasoning effort"
        " and the second at low. The new request must hold the whole first turn;"
        " `request.valid` judges its reasoning."
    )
    needs = ("continue", "effort")

    def first(self):
        return {"settings": {"effort": "high"}}

    def second(self):
        return {"settings": {"effort": "low"}}

    @check(REQUIRED)
    def complete(self, run):
        return continued(run)


class SwitchApi(TwoTurns):
    id = "session.switch-api"
    title = "Continuing on another wire API converts the history"
    description = (
        "The first turn runs over one API and the session continues over another, as"
        " when a user switches provider. The history must arrive complete;"
        " `request.valid` judges whether the second API accepts its tool IDs,"
        " results and reasoning."
    )
    variants = tuple(PAIRS)
    per_harness = True
    needs = ("continue", "two-apis")

    def unsupported(self):
        missing = [api for api in PAIRS[self.variant] if api not in self.recipe.apis]
        return f"The recipe does not configure {', '.join(missing)}" if missing else None

    def first(self):
        return {"api": PAIRS[self.variant][0]}

    def second(self):
        return {"api": PAIRS[self.variant][1]}

    @check(REQUIRED)
    def complete(self, run):
        return continued(run)

    @check(
        INFO,
        id="reasoning.across-apis",
        title="Where earlier reasoning goes after a switch to another API",
        description=(
            "In the same switch of wire API. Anthropic puts another provider's turn in"
            ' as text and tool calls only, so "dropped" is the documented way;'
            " reasoning as visible text or in a user message is discouraged."
        ),
    )
    def reasoning(self, run):
        return reasoning_place(run)


class Crash(Scenario):
    id = "session.crash"
    title = "A session resumes after the agent is killed"
    description = (
        "The agent is killed with SIGKILL in the middle of a turn, after one tool"
        " call finished and as a second one starts, as a crash or a closed laptop"
        " would end it. Then the user continues the session. The agent must start"
        " and send a request; losing the unfinished turn is allowed."
    )
    tools = True
    needs = ("continue",)

    def launches(self):
        return [
            Launch(prompt=f"{PROMPT}\n{ONE}", kill_when=slow_started, timeout=90),
            Launch(prompt=f"{PROMPT}\n{TWO}", resume=True, track=1),
        ]

    def scripts(self):
        return [[use("text"), use(Use("slow", seconds=30)), say(FIRST_DONE)], [say(SECOND_DONE)]]

    @check(REQUIRED)
    def resumes(self, run):
        if run.launches[0].termination != "killed":
            return skipped("The suite never killed the agent.")
        # Only the killed launch: a resumed agent may answer the call itself.
        slow = {call.id for served in run.served if served.step == 1 for call in served.calls}
        if any(e.launch == 0 and any(r.id in slow for r in e.request.results) for e in run.tasks):
            return skipped("The second call ended before the kill.")
        request = second_request(run)
        if request is None:
            return failed("no second turn", f"The resumed agent sent no request. {run.exit_detail()}".strip())
        return passed()


LONG_STEPS = 40
LONG_RESULT = 4096  # Bytes of each tool result: 1024 tokens.


class Long(Scenario):
    id = "session.long"
    title = "A long tool loop finishes"
    description = (
        f"The model calls the tool {LONG_STEPS} times in one turn, each result"
        f" {LONG_RESULT // 1024} KB of text, far below the context window. The agent"
        " must run every call once and finish the turn: a step limit that stops it"
        " is a gap."
    )
    tools = True
    timeout = 300.0
    max_requests = LONG_STEPS + 20

    def script(self):
        return [*(use(Use("text", size=LONG_RESULT, style="lines")) for _ in range(LONG_STEPS)), DONE]

    @check(RECOMMENDED)
    def finishes(self, run):
        ran = len({c["arguments"].get("token") for c in run.fixture_calls})
        if run.finished_script and ran >= LONG_STEPS:
            return passed()
        return gap(f"stopped after {ran}", f"{ran} of {LONG_STEPS} calls ran. {run.exit_detail()}".strip())
