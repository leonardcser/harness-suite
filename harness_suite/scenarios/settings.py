"""Settings: what the agent's own options change in the request."""

from ..apis.anthropic import thinking
from ..results import OPTIONAL, RECOMMENDED, check, failed, gap, passed, skipped, unsupported
from ..scenario import Scenario
from ..script import DONE, PROMPT, Launch, Use, say, use


def reasoning_settings(request) -> dict:
    keys = ("effort", "thinking", "thinking_budget", "max_tokens", "service_tier", "speed")
    values = {k: request.parameters[k] for k in keys if k in request.parameters}
    if request.parameters.get("beta"):
        values["beta"] = request.parameters["beta"]
    return values


def first_per_launch(run) -> list:
    found = {}
    for exchange in run.tasks:
        found.setdefault(exchange.launch, exchange.request)
    return [found.get(index) for index in range(len(run.launches))]


class Effort(Scenario):
    id = "settings.effort"
    title = "The reasoning effort setting changes the request"
    description = (
        "The harness starts once with low and once with high reasoning effort. The"
        " requests should differ in the API's reasoning field: `reasoning_effort`,"
        " `reasoning.effort`, `output_config.effort` or a thinking budget."
    )
    needs = ("effort",)

    def launches(self):
        return [Launch(prompt=PROMPT, settings={"effort": "low"}), Launch(prompt=PROMPT, settings={"effort": "high"})]

    def script(self):
        return [say("SUITE_FIRST"), DONE]

    @check(OPTIONAL)
    def changes(self, run):
        low, high = first_per_launch(run)
        if not low or not high:
            return skipped(f"No request reached the provider. {run.exit_detail()}".strip())
        a, b = reasoning_settings(low), reasoning_settings(high)
        if a == b:
            # The recipe configures the setting, and tells the agent what its
            # model supports, so a setting with no effect fails.
            return failed("no change", f"Both send {a}: the setting does not change the request.")
        return passed(detail=f"low: {a}; high: {b}")


class Temperature(Scenario):
    id = "settings.temperature"
    title = "A temperature from the configuration reaches the request"
    description = (
        "The agent is configured with a sampling temperature of 0.3, in its own"
        " way: a flag, or its configuration file. The request should carry it as"
        " `temperature`. Anthropic takes only 1 while thinking is on, so there an"
        " agent may leave it out."
    )
    needs = ("temperature",)

    def launches(self):
        return [Launch(settings={"temperature": 0.3})]

    def script(self):
        return [say("SUITE_FIRST"), DONE]

    @check(OPTIONAL)
    def reaches(self, run):
        if not run.requests:
            return skipped(f"No request reached the provider. {run.exit_detail()}".strip())
        request = run.requests[0]
        sent = request.parameters.get("temperature")
        if run.api == "anthropic" and thinking(request.raw) and sent in (None, 1):
            how = "left out" if sent is None else "sent 1"
            return passed(how, "Anthropic takes only a temperature of 1 while thinking is on, as here.")
        if sent is None:
            return failed("not sent", "The agent took the setting, but no request carries a temperature.")
        return passed() if abs(float(sent) - 0.3) < 1e-9 else failed(f"sent {sent}")


SYSTEM_RULE = "SUITE_SYSTEM_RULE: state the file names you change."


class SystemPrompt(Scenario):
    id = "settings.system-prompt"
    title = "A system prompt from the configuration reaches every request"
    description = (
        "The agent is configured with a system prompt of its user's, in its own"
        " way: a flag that adds to its prompt or replaces it, or its configuration."
        " Every request of the tool loop must carry that text as instructions:"
        " the system prompt, or a system or developer message."
    )
    needs = ("system-prompt",)
    tools = True
    sources = ("native",)

    def launches(self):
        return [Launch(settings={"system-prompt": SYSTEM_RULE})]

    def script(self):
        return [use("text"), DONE]

    @check(OPTIONAL)
    def reaches(self, run):
        if not run.requests:
            return skipped(f"No request reached the provider. {run.exit_detail()}".strip())
        found = [SYSTEM_RULE in r.system for r in run.requests]
        if all(found):
            own = len(run.requests[0].system) - len(SYSTEM_RULE)
            return passed(detail=f"The system prompt holds it, and {own} characters besides.")
        if any(SYSTEM_RULE in text for r in run.requests for text in r.user_texts):
            return failed("as user text", "The text came in a user message, not as instructions.")
        if any(found):
            return failed("dropped later", f"{found.count(False)} of {len(found)} requests lack it.")
        return failed("not sent", "The agent took the setting, but no request carries the text.")


HEADER, HEADER_VALUE = "x-suite-header", "suite-gateway-7f3a"


class Headers(Scenario):
    id = "settings.headers"
    title = "A custom header from the configuration reaches every request"
    description = (
        f"The agent is configured, in its own way, to send the header"
        f" `{HEADER}: {HEADER_VALUE}`, as a gateway that routes or bills by a"
        " header needs. Every request to the provider, side requests too, must"
        " carry it: a gateway rejects the ones that do not."
    )
    needs = ("headers",)
    tools = True
    sources = ("native",)

    def launches(self):
        return [Launch(settings={"headers": HEADER_VALUE})]

    def script(self):
        return [use("text"), DONE]

    @check(OPTIONAL)
    def reaches(self, run):
        requests = [e.request for e in run.exchanges if e.request]  # Side requests too.
        if not requests:
            return skipped(f"No request reached the provider. {run.exit_detail()}".strip())
        sent = [{k.lower(): v.strip() for k, v in r.headers.items()}.get(HEADER) for r in requests]
        if all(value == HEADER_VALUE for value in sent):
            return passed(detail=f"On all {len(sent)} requests.")
        if any(value == HEADER_VALUE for value in sent):
            missing = sum(value != HEADER_VALUE for value in sent)
            return failed("missing on some", f"{missing} of {len(sent)} requests lack it.")
        if any(sent):
            return failed("wrong value", f"Sent {next(v for v in sent if v)!r}.")
        return failed("not sent", "The agent took the setting, but no request carries the header.")


READ_ONLY_FILE = "suite-read-only.txt"
READ_ONLY_KEPT = "SUITE_READ_ONLY: the file as the user left it.\n"
READ_ONLY_TRIES = {
    "write": Use("write", value=READ_ONLY_FILE, content="SUITE_CHANGED by the write tool.\n"),
    "shell": Use("text", command=f"printf 'SUITE_CHANGED by the shell.\\n' > {READ_ONLY_FILE}"),
}


class ReadOnly(Scenario):
    id = "settings.read-only"
    title = "A read-only mode changes no file"
    description = (
        "The agent runs in its read-only or planning mode, or with only its"
        " read-only tools. The model tries to overwrite a workspace file with the"
        " agent's own write tool (`write`), or with a shell redirection (`shell`)."
        " The file must stay as it was: the agent may leave the tool out, refuse"
        " the call, or run the command where it cannot write."
    )
    needs = ("read-only",)
    tools = True
    sources = ("native",)
    variants = tuple(READ_ONLY_TRIES)
    per_harness = True

    @classmethod
    def bound(cls, variant):
        return ("write",) if variant == "write" else ()

    def files(self):
        return {READ_ONLY_FILE: READ_ONLY_KEPT.encode()}

    def launches(self):
        return [Launch(settings={"read-only": True})]

    def script(self):
        return [use(READ_ONLY_TRIES[self.variant]), DONE]

    def outputs(self):
        return [READ_ONLY_FILE]

    @check(OPTIONAL)
    def kept(self, run):
        data = run.outputs.get(READ_ONLY_FILE)
        if data != READ_ONLY_KEPT.encode():
            return failed("file changed", f"The file now holds {data[:80]!r}." if data else "The file is gone.")
        if not run.served:
            return skipped(f"The agent sent no request. {run.exit_detail()}".strip())
        name = self.recipe.native.get("write" if self.variant == "write" else "shell", {}).get("name")
        if name and run.requests and all(t.name != name for t in run.requests[0].tools):
            return passed("tool left out", f"The agent does not offer `{name}` in this mode.")
        results = run.step_result(0)
        said = f" The result: {results[0].text[:160]!r}" if results else " No result came back."
        return passed(detail="The call did not change the file." + said)


class BrokenFile(Scenario):
    id = "settings.broken-file"
    title = "A broken configuration file is reported"
    description = (
        "The agent first does a task with its normal configuration, as a control."
        " Then the first file the recipe writes, the agent's configuration, ends"
        " with a line that no format parses, as after a bad edit or merge, and the"
        " agent starts again. It should stop with a status that is not 0, or go"
        " on and do the task; it must not hang, nor exit 0 without doing"
        " anything, which tells a script that it worked. The suite does not read"
        " what the agent writes about it. A recipe that configures the agent"
        " without files is not measured, nor an agent that fails the control."
    )
    per_harness = True
    timeout = 90.0
    silent = True  # It may stop before any request.

    def launches(self):
        return [Launch(), Launch(broken_file=True)]

    def script(self):
        return [DONE, DONE]

    @check(RECOMMENDED)
    def reported(self, run):
        control, launch = run.launches
        if launch.broken is None:
            return unsupported(detail="The recipe writes no configuration file.")
        if not any(e.launch == 0 for e in run.tasks) or control.exit != 0:
            return skipped(f"With its normal configuration, the agent did not do the task. {run.exit_detail()}".strip())
        name = launch.broken.rsplit("/", 1)[-1]
        if any(e.launch == 1 for e in run.tasks):
            return passed("went on", f"It did the task with a broken {name}.")
        if not run.ended or launch.exit is None:
            return gap("no exit", f"Still running with a broken {name} when the suite stopped it.")
        if launch.exit == 0:
            return gap("exit 0", f"It did nothing, and said it worked, with a broken {name}.")
        return passed(f"stopped, exit {launch.exit}")
