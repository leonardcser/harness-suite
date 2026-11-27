"""The agent's own shell tool: how it runs commands that behave as real ones
do. The model calls the fixture through the agent's shell tool, so each
command is a real process that the agent starts and reads."""

import re

from ..fixture import EXIT_STATUS, marker
from ..results import INFO, RECOMMENDED, REQUIRED, check, failed, gap, label, passed, skipped
from ..scenario import Scenario
from ..script import DONE, Use, use
from .tools import kept

NOT_RUN = "The command never ran; tools.round-trip judges whether commands run."


class Command(Scenario):
    """One shell command, then the end of the task."""

    tools = True
    sources = ("native",)
    kind = ""

    def script(self):
        return [use(self.kind), DONE]

    def token(self, run) -> str | None:
        return run.tokens(0)[0] if run.served else None

    def result(self, run):
        results = run.step_result(0)
        return results[0] if results else None

    def arrival(self, run) -> float | None:
        """When the request that carried the command's result started."""
        call = run.calls[0] if run.calls else None
        return next((e.started for e in run.tasks if call and e.request.result(call.id)), None)


class Binary(Command):
    id = "shell.binary"
    title = "Output that is not UTF-8 comes back"
    description = (
        "The command prints bytes that are not valid UTF-8, as `cat` of a binary"
        " file or a program in another locale does. The agent must not crash: the"
        " result must come back with the text around the bad bytes."
    )
    kind = "binary"

    @check(REQUIRED)
    def returned(self, run):
        if not run.served:
            return skipped(NOT_RUN)
        result, token = self.result(run), self.token(run)
        if result is None:
            return failed("no result", f"The result never reached the model. {run.exit_detail()}".strip())
        if marker(token) not in result.text or f"SUITE_END_{token}" not in result.text:
            return failed("clipped", f"The result lost text around the bad bytes: {result.text[:120]!r}")
        return passed()


class Output(Command):
    id = "shell.output"
    title = "A failing command's output and errors both come back"
    description = (
        "The command prints a line on stdout, another on stderr, and exits with"
        f" status {EXIT_STATUS}, as a failing test run does. The model needs both lines to"
        " understand what went wrong."
    )
    kind = "mixed"

    @check(REQUIRED)
    def both(self, run):
        if not run.served:
            return skipped(NOT_RUN)
        result, token = self.result(run), self.token(run)
        if result is None:
            return failed("no result")
        missing = [
            name
            for name, text in (("stdout", marker(token)), ("stderr", f"SUITE_STDERR_{token}"))
            if text not in result.text
        ]
        if missing:
            return failed(f"lost {' and '.join(missing)}", f"The result: {result.text[:160]!r}")
        return passed()

    @check(
        RECOMMENDED,
        id="shell.exit-status",
        title="The model can tell that a command failed",
        description=(
            f"In the same failing command, the result should hold the exit status {EXIT_STATUS},"
            " a number no other part of the output has, or be marked as an error."
            " Without either, a failing command that prints nothing looks like one"
            " that worked."
        ),
    )
    def status(self, run):
        result = self.result(run)
        if result is None:
            return skipped("No result came back; shell.output judges that.")
        if re.search(rf"\b{EXIT_STATUS}\b", result.text.replace(self.token(run) or "", "")):
            return passed("status shown")
        if result.error:
            return passed("error flag")
        return gap("status hidden", f"The result: {result.text[:160]!r}")


class Background(Command):
    id = "shell.background"
    title = "A command that leaves a background process returns at once"
    description = (
        "The command starts a child process and exits, as `server &` or a build"
        " watcher does. The child keeps the command's output open for 30 s. An"
        " agent that waits for the output to close, instead of for the command to"
        " exit, hangs as long as the child lives."
    )
    kind = "background"
    timeout = 90.0

    @check(RECOMMENDED)
    def returned(self, run):
        token = self.token(run)
        end = run.fixture_events("end", token) if token else []
        arrived = self.arrival(run)
        if not end:
            return skipped(NOT_RUN)
        if arrived is None:
            return gap("no result", "The result never reached the model.")
        waited = arrived - run.clock(end[0]["time"])
        if waited > 10:
            return gap("waited for the child", f"The agent waited {waited:.0f} s for the child to close the output.")
        return passed("returned", f"The result came {max(waited, 0):.0f} s after the command ended.")


class Inspect(Command):
    id = "shell.cwd"
    title = "Commands run in the workspace"
    description = (
        "The command reports its working directory. A coding agent should run"
        " commands in the workspace it was started in, so relative paths work."
    )
    kind = "inspect"

    @check(RECOMMENDED)
    def workspace(self, run):
        token = self.token(run)
        seen = run.fixture_events("inspect", token) if token else []
        if not seen:
            return skipped(NOT_RUN)
        cwd = seen[0]["cwd"]
        if cwd.rstrip("/") == run.workspace.rstrip("/"):
            return passed()
        return gap("elsewhere", f"The command ran in {cwd}, not {run.workspace}.")


class Stdin(Command):
    id = "shell.stdin"
    title = "A command that reads its input does not hang"
    description = (
        "The model runs a shell command that reads standard input to its end, as"
        " many scripts and tools do when they get no file. The agent should give"
        " commands an empty, closed input, so the command ends at once. An agent"
        " that leaves the input open hangs until its tool timeout."
    )
    kind = "stdin"
    timeout = 90.0

    @check(RECOMMENDED)
    def closed(self, run):
        token = self.token(run)
        if token is None:
            return skipped(NOT_RUN)
        start = run.invocations(token)
        end = run.fixture_events("end", token)
        if start and end:
            took = end[0]["time"] - start[0]["time"]
            if took < 10:
                return passed("input closed", f"The command read {end[0]['stdin']} bytes and ended.")
            return gap("input closed late", f"The command's input closed after {took:.0f} s.")
        if run.step_result(0):
            return gap("timed out", "The command still waited on its input when the agent gave up on it.")
        return gap("hung", "The command was still waiting on its input when the suite stopped.")


CHILD_GRACE = 2.0  # Seconds a stopped command's child may still beat after the result.
WATCH = 8.0  # Seconds of the next command, while the suite watches for a child left behind.


class Timeout(Command):
    id = "shell.timeout"
    title = "A command that never ends gets a result, and its child is not orphaned"
    description = (
        "The command runs for 10 minutes, its work in a child process, as a hung"
        " test run would. The agent should give the model a result within the"
        " 200 s the suite waits: stop the command, or move it to the background"
        " and still hold it. A child that the agent orphaned, by killing only the"
        " command's own process, runs on unseen: the fixture marks its beats once"
        f" its parent is gone. The model then runs a command of {WATCH:g} s, while"
        " the suite watches for those beats. The detail says after how long the"
        " result came."
    )
    timeout = 200.0

    def script(self):
        # The next command keeps the agent running while the suite watches
        # for beats of a child left behind: the agent ends every process it
        # still holds when it exits.
        return [use(Use("slow", seconds=600)), use(Use("slow", seconds=WATCH)), DONE]

    @check(RECOMMENDED)
    def stopped(self, run):
        token = self.token(run)
        start = run.invocations(token) if token else []
        if not start:
            return skipped(NOT_RUN)
        began = run.clock(start[0]["time"])
        arrived = self.arrival(run)
        if arrived is None:
            return gap("no result", f"No result came back in the {self.timeout:.0f} s the suite watched.")
        watched = next(
            (e.started for e in run.tasks if run.was_served(1) and e.request.result(run.served[1].calls[0].id)), None
        )
        if watched is None:
            return skipped(
                "The agent did not run the next command, so the suite could not watch for a child left behind."
            )
        orphaned = [
            t
            for b in run.fixture_events("beat", token)
            if b.get("orphan") and arrived + CHILD_GRACE < (t := run.clock(b["time"])) <= watched
        ]
        if orphaned:
            return gap(
                "child orphaned",
                f"The child ran on without its parent {orphaned[-1] - arrived:.0f} s after the result.",
            )
        return passed("stopped", f"The result came {arrived - began:.0f} s after the command started.")


class Terminal(Command):
    id = "shell.terminal"
    title = "A command cannot wait on the user's terminal"
    description = (
        "The command asks yes or no on its controlling terminal, as `ssh`, `sudo`"
        " or a package manager do (`prompt`), or prints a long log into a pager"
        " when its output is a terminal, as `git log` does (`pager`). No one can"
        " answer: the model cannot type, and in the TUI the keys belong to the"
        " agent. The agent should run commands without a terminal, or with the"
        " pager off, so they end at once. Otherwise the command waits 20 s, and in"
        " the TUI it can read the user's keys."
    )
    variants = ("prompt", "pager")
    timeout = 90.0

    @property
    def kind(self) -> str:
        return self.variant

    @check(RECOMMENDED)
    def detached(self, run):
        token = self.token(run)
        seen = run.fixture_events(self.variant, token) if token else []
        if not seen:
            return skipped(NOT_RUN)
        if self.variant == "prompt" and seen[0]["tty"]:
            return gap("terminal open", "The command opened /dev/tty and waited for an answer.")
        if self.variant == "pager" and seen[0]["paged"]:
            return gap("pager waited", f"The output was a terminal, and the pager {seen[0]['pager']!r} waited.")
        if not run.step_result(0):
            return gap("no result")
        if self.variant == "pager" and seen[0]["tty"]:
            return passed("pager off", f"The output was a terminal, with the pager {seen[0]['pager']!r}.")
        return passed("no terminal")


class Flood(Command):
    id = "shell.flood"
    title = "A command that prints a million lines"
    description = (
        "The command prints a million short lines, 27 MB, as a verbose build or"
        " `find /` does. The agent should stay up, keep its requests within the"
        " context window, and give the model a part of the output. In `failing`,"
        " the command then prints why it failed, as its last line, and exits. A pass says"
        " how much output reached the model, and which end."
    )
    kind = "flood"
    window = 200_000
    variants = ("default", "failing")

    def script(self):
        style = "failing" if self.variant == "failing" else None
        return [use(Use("flood", style=style)), DONE]

    @check(RECOMMENDED)
    def survived(self, run):
        if not run.served:
            return skipped(NOT_RUN)
        if any(e.kind == "over-window" for e in run.exchanges):
            return gap("over the window", "A request with the output was larger than the context window.")
        results = run.step_result(0, anywhere=True)
        if not results:
            return gap("no result", f"The output never reached the model. {run.exit_detail()}".strip())
        if not run.finished_script:
            return gap("stopped", "The agent did not go on after the command.")
        return passed(*kept(results[0].text, self.token(run)))

    @check(
        RECOMMENDED,
        id="shell.flood-status",
        title="A failing command with a flood of output still shows why it failed",
        description=(
            "In `failing`, after its million lines, the command prints why it failed"
            " in a last line that the suite marks. However much of the output the"
            " agent cuts, the model should still see that last line."
        ),
        variants=("failing",),
    )
    def status(self, run):
        if not run.served:
            return skipped(NOT_RUN)
        result, token = self.result(run), self.token(run)
        if result is None:
            return gap("no result", f"The output never reached the model. {run.exit_detail()}".strip())
        if f"SUITE_FLOOD_ERROR_{token}" not in result.text:
            return gap("lost the last line", f"The model got {len(result.text.encode()):,} bytes.")
        return passed()


LONG_COMMAND = 200_000  # Characters: over Linux's 128 KB limit on one argument.


class LongCommand(Command):
    id = "shell.long-command"
    title = "A very long command line runs or gets a result"
    description = (
        "The model writes a command of 200 KB, as when it inlines a file into a"
        " heredoc. Linux limits one argument of a process to 128 KB, so an agent"
        " that passes the command as `bash -c COMMAND` cannot run it, while one"
        " that writes it to the shell's input can. The command should run whole,"
        " or come back as a result, such as an error, so the model can try"
        " another way. A command cut short, or no result, is a gap."
    )
    kind = "long"

    def script(self):
        return [use(Use("long", value="x" * LONG_COMMAND, heredoc=True)), DONE]

    @check(RECOMMENDED)
    def ran(self, run):
        if not run.served:
            return skipped(NOT_RUN)
        token = self.token(run)
        end = run.fixture_events("end", token)
        result = self.result(run)
        if end and end[0]["length"] == LONG_COMMAND:
            return passed("ran")
        if end:
            return gap("changed", f"The command received {end[0]['length']} of {LONG_COMMAND} chars.")
        if cut := run.fixture_events("unparsed", token):
            return gap("changed", f"The command reached the fixture cut short, at {cut[0]['length']} chars.")
        if result is None:
            return gap("no result", f"The agent gave no result for the command. {run.exit_detail()}".strip())
        return passed("result, not run", f"The result: {result.text[:160]!r}")


class ChangeDirectory(Command):
    id = "shell.cd"
    title = "Whether a `cd` carries over to the next command"
    description = (
        "The model runs `mkdir -p suite_sub && cd suite_sub`, then a command that"
        " reports its working directory. Some agents keep one shell for the"
        " session, so the second command runs in the new directory; others start"
        " each command afresh in the workspace. Models act on one or the other."
    )

    def script(self):
        return [use(Use("text", command="mkdir -p suite_sub && cd suite_sub")), use("inspect"), DONE]

    def token(self, run) -> str | None:
        tokens = run.tokens(1) if run.served else []
        return tokens[0] if tokens else None

    @check(INFO)
    def kept(self, run):
        token = self.token(run)
        seen = run.fixture_events("inspect", token) if token else []
        if not seen:
            return skipped("The command after the cd never ran.")
        cwd = seen[0]["cwd"].rstrip("/")
        if cwd.endswith("/suite_sub"):
            return label("kept")
        if cwd == run.workspace.rstrip("/"):
            return label("reset to workspace")
        return label("elsewhere", f"The command ran in {cwd}.")
