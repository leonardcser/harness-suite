"""The command line contract of headless mode, which scripts and CI jobs rely
on: exit status, input and output."""

import json
import re

from ..evidence import KEPT_OUTPUT
from ..messages import Error
from ..results import INFO, RECOMMENDED, REQUIRED, check, failed, gap, label, passed, skipped
from ..scenario import Scenario
from ..script import DONE, Launch, Step, fail_with, say, use


class ExitStatus(Scenario):
    id = "cli.exit-status"
    title = "The exit status tells a script whether the task worked"
    description = (
        "Headless, the agent finishes a task (`success`), finishes it after a"
        " rate limit that a retry got past (`recovered`), meets an invalid API key"
        " on every request (`failure`), or meets it after a tool call already ran"
        " (`failure-after-tool`). A script that runs the agent can only tell"
        " success from failure by the exit status: 0 after success, not 0 after a"
        " failure. The recipe picks the agent's own headless output mode."
    )
    tools = True
    variants = ("success", "recovered", "failure", "failure-after-tool")
    timeout = 90.0
    max_requests = 100

    @classmethod
    def uses_tools(cls, variant: str) -> bool:
        return variant == "failure-after-tool"

    def script(self):
        forever = Step(error=Error("auth"), times=10_000)
        return {
            "success": [DONE],
            "recovered": [fail_with("rate", retry_after="1"), DONE],
            "failure": [forever, DONE],
            "failure-after-tool": [use("text"), forever, DONE],
        }[self.variant]

    @check(RECOMMENDED)
    def status(self, run):
        if not run.ended or run.last.exit is None:
            return gap(
                "no exit", f"The agent was still running when the suite stopped it, after {len(run.tasks)} requests."
            )
        worked = self.variant in ("success", "recovered")
        if worked and not run.finished_script:
            return skipped("The task did not finish.")
        if not worked and not any(e.error for e in run.tasks if not self.uses_tools(self.variant) or run.fixture_calls):
            return skipped(
                "The agent never met the error" + (" after the tool ran." if self.uses_tools(self.variant) else ".")
            )
        code = run.last.exit
        if (code == 0) == worked:
            return passed(f"exit {code}")
        return gap(f"exit {code}")


def parses(text: str) -> bool:
    try:
        json.loads(text)
    except ValueError:
        return False
    return True


class Json(Scenario):
    id = "cli.json"
    title = "Machine-readable output is valid JSON"
    description = (
        "Headless, in the JSON output mode of the recipe's `json` setting, the"
        " agent answers (`text`), runs a tool first (`tools`), or meets an invalid"
        " API key on every request (`failure`). All it writes to stdout should"
        " parse: one JSON document, or one JSON value per line. A log line or a"
        " progress message among them breaks every script that reads the output."
        " In `failure`, the exit status must also not be 0: a script that reads"
        " JSON relies on it as much as one that reads text."
    )
    variants = ("text", "tools", "failure")
    needs = ("json",)
    timeout = 90.0
    max_requests = 100

    @classmethod
    def uses_tools(cls, variant):
        return variant == "tools"

    def launches(self):
        return [Launch(settings={"json": True})]

    def script(self):
        if self.variant == "failure":
            return [Step(error=Error("auth"), times=10_000), DONE]
        if self.variant == "tools":
            return [use("text"), DONE]
        return [DONE]

    @check(RECOMMENDED)
    def parses(self, run):
        out = run.launches[0].stdout
        if len(out) > KEPT_OUTPUT:
            return skipped(f"Over {KEPT_OUTPUT // 1024**2} MB of output.")
        if not out.strip():
            return gap("no output")
        values = [line for line in out.splitlines() if line.strip()]
        bad = next((n for n, line in enumerate(values) if not parses(line)), None)
        if bad is not None and not parses(out):
            return gap("not JSON", f"Line {bad + 1} of {len(values)}: {values[bad].strip()[:160]}")
        launch = run.launches[0]
        if self.variant == "failure" and run.ended and launch.exit == 0:
            return gap("exit 0", "The output is JSON, but the failed run exited with status 0.")
        return passed("JSON lines" if bad is None else "one document", f"{len(values)} values.")


UNKNOWN_OPTION = "--suite-no-such-option"


def control(run) -> str | None:
    """Why the first launch, a normal task, does not show that the agent works
    headless, so that what the second launch did not do says nothing."""
    if 0 not in run.finished:
        return f"The normal launch did not finish its task. {run.ending(0)}"
    if run.launches[0].exit != 0:
        return f"The normal launch exited with {run.launches[0].exit}."
    if len(run.launches) < 2:
        return run.ending(1)
    return None


class Controlled(Scenario):
    """A normal launch, then the launch under test on its own track."""

    silent = True
    per_harness = True

    def launches(self):
        return [Launch(timeout=60.0), self.tested()]

    def tested(self) -> Launch:
        raise NotImplementedError

    def scripts(self):
        return [[DONE], [DONE]]


class UsageError(Controlled):
    id = "cli.usage-error"
    title = "An unknown option fails at once"
    description = (
        f"The agent starts with `{UNKNOWN_OPTION}`, an option it does not have, as"
        " a typo or a script written for another version gives it. It should exit"
        " with a status that is not 0, without starting the task: a script must"
        " not take a run it got wrong for one that worked. A normal launch first"
        " shows that the agent works headless."
    )
    timeout = 30.0

    def tested(self):
        return Launch(options=(UNKNOWN_OPTION,), track=1)

    @check(RECOMMENDED)
    def refused(self, run):
        if problem := control(run):
            return skipped(problem)
        launch = run.launches[1]
        if not run.ended or launch.exit is None:
            return gap("no exit", "The agent was still running when the suite stopped it.")
        if sent := [e for e in run.exchanges if e.launch == 1 and e.request]:
            return gap("ran the task", f"It sent {len(sent)} requests, then exited with {launch.exit}.")
        if launch.exit == 0:
            return gap("exit 0", "It did nothing, and said it worked.")
        return passed(f"exit {launch.exit}")


class EmptyPrompt(Controlled):
    id = "cli.empty-prompt"
    title = "An empty prompt fails at once"
    description = (
        "The agent gets an empty prompt, as from a script whose variable came out"
        " empty, or one of only spaces and new lines (`whitespace`). It should exit"
        " at once with a status that is not 0, and send no request: a blank user"
        " message is no task, and Anthropic rejects one. A normal launch first"
        " shows that the agent works headless."
    )
    timeout = 30.0
    variants = ("empty", "whitespace")

    def tested(self):
        return Launch(prompt="" if self.variant == "empty" else "  \n\t\n", track=1)

    @check(RECOMMENDED)
    def refused(self, run):
        if problem := control(run):
            return skipped(problem)
        launch = run.launches[1]
        if not run.ended or launch.exit is None:
            return gap("no exit", "The agent was still running when the suite stopped it.")
        if sent := [e for e in run.exchanges if e.launch == 1 and e.request]:
            return gap("sent a request", f"{len(sent)} requests, then exit {launch.exit}.")
        if launch.exit == 0:
            return gap("exit 0", "It did nothing, and said it worked.")
        return passed(f"exit {launch.exit}")


class OpenStdin(Controlled):
    id = "cli.open-stdin"
    title = "Whether a prompt in the arguments waits for an open stdin"
    description = (
        "The prompt is in the arguments, and stdin is a pipe that stays open and"
        " sends nothing, as in a `while read` loop, over SSH, or from a parent"
        " process that keeps the pipe it started the agent with. Some agents read"
        ' a piped stdin to add it to the prompt, as `cat log | agent "explain"`'
        " needs, and so wait for it to close; others ignore stdin when the"
        " arguments hold the prompt. The label shows which. A normal launch first"
        " shows that the agent works headless."
    )
    timeout = 60.0

    def tested(self):
        return Launch(open_stdin=True, track=1)

    @check(INFO)
    def goes_on(self, run):
        if problem := control(run):
            return skipped(problem)
        sent = [e for e in run.tasks if e.launch == 1]
        if not sent:
            if run.ended and run.launches[1].exit is not None:
                return skipped(f"It exited with no request. {run.exit_detail()}".strip())
            return label("waits for stdin", f"No request in {self.timeout:.0f} s, while stdin stayed open.")
        wait = sent[0].started - run.launches[1].started
        return label("ignores stdin", f"First request {wait:.1f} s after the start.")


# A control sequence (CSI), an operating system command (OSC), or another escape.
ESCAPE = re.compile(r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)|[0-9=>@-_])")

ANSWERS = {
    "unicode": (
        "SUITE_ANSWER_START\n"
        "Accents: café, naïve. CJK: 漢字かな. Emoji: 🎉👍🏽. Right to left: שלום.\n"
        "A tab:\there. Quotes: \"double\" 'single' `back`. A backslash: \\ and \\n as text.\n"
        "SUITE_ANSWER_END"
    ),
    "long": "".join(f"SUITE_LINE_{n:05d} {'abcdefghij' * 6}\n" for n in range(1500)) + "SUITE_ANSWER_END",
}


def strings(value) -> list[str]:
    """Every string in a JSON value."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [s for item in value for s in strings(item)]
    if isinstance(value, dict):
        return [s for item in value.values() for s in strings(item)]
    return []


def printed(out: str, answer: str) -> bool:
    """Whether stdout holds the answer whole: as text, or as one string of its JSON."""
    if answer in out:
        return True
    documents = [out, *out.splitlines()]
    for document in documents:
        try:
            if any(answer in text for text in strings(json.loads(document))):
                return True
        except ValueError:
            continue
    return False


class Answer(Scenario):
    id = "cli.answer"
    title = "The final answer reaches stdout whole"
    description = (
        "Headless, the model's final answer is the agent's output: a script reads"
        " it from stdout, as text or as a string in the recipe's JSON output. The"
        " answer has accents, CJK, emoji, right-to-left text, tabs, quotes and"
        " backslashes (`unicode`), or is 100 KB of lines (`long`), which an agent"
        " that exits before it flushes a pipe cuts short. Every character must"
        " arrive unchanged."
    )
    variants = tuple(ANSWERS)
    timeout = 90.0

    def script(self):
        return [say(ANSWERS[self.variant])]

    @check(REQUIRED)
    def whole(self, run):
        if not run.finished_script:
            return skipped(f"The model's answer was never served. {run.exit_detail()}".strip())
        out = run.launches[0].stdout
        answer = ANSWERS[self.variant]
        if printed(out, answer):
            return passed()
        start = answer.splitlines()[0]
        if printed(out, start):
            return failed("cut or changed", f"The answer starts in the output, but not whole: {len(out)} characters.")
        return failed("changed", "The answer is not in the output as the model wrote it.")

    @check(
        RECOMMENDED,
        id="cli.no-escapes",
        title="Output to a pipe holds no terminal escape codes",
        description=(
            "In the same run, stdout is a pipe, not a terminal: colors, cursor"
            " moves and spinners there end up in the files and logs of every"
            " script that reads it."
        ),
    )
    def plain(self, run):
        out = run.launches[0].stdout
        if not out.strip():
            return skipped("The agent printed nothing.")
        found = ESCAPE.findall(out)
        if found:
            shown = ", ".join(dict.fromkeys(repr(code) for code in found[:5]))
            return gap("escape codes", f"{len(found)} in the output, such as {shown}.")
        return passed()
