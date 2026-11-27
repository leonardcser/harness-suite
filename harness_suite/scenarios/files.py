"""The agent's own file tools, with files that real workspaces have: binary,
huge, one endless line, missing, or not UTF-8, and writes and edits that must
keep every byte. The recipe binds the tools as `[native.read]`, `[native.write]`
and `[native.edit]`."""

import contextlib
import os
import random
import re
import threading
from pathlib import Path

from ..results import INFO, RECOMMENDED, check, described, gap, label, passed, skipped
from ..scenario import Scenario
from ..script import DONE, Use, use

MARK, END = "SUITE_FILE_START", "SUITE_FILE_END"


def binary() -> bytes:
    """A compiled file: a header, NUL bytes, bytes of every value, and 4 KB
    that look random but are the same in every run, between the markers."""
    noise = random.Random(0).randbytes(4096)
    body = MARK.encode() + bytes(range(256)) * 64 + noise + END.encode()
    return b"\x7fELF\x02\x01\x01\0" + b"\0" * 56 + body


def large() -> bytes:
    """About 5 MB of numbered lines."""
    lines = [f"{MARK}\n"] + [f"line {n:07d}: the quick brown fox jumps over the lazy dog\n" for n in range(80_000)]
    return ("".join(lines) + END + "\n").encode()


def long_line() -> bytes:
    """One line of 1 MB, as minified JavaScript or a data dump has."""
    return (MARK + "x" * 1_000_000 + END + "\n").encode()


def latin1() -> bytes:
    return f"{MARK}\ncafé à l'élève, naïve façade\n{END}\n".encode("latin-1")


def directory(path: Path) -> None:
    path.mkdir(parents=True)
    (path / "inside.txt").write_text(MARK)


def fifo(path: Path) -> None:
    """A named pipe that no one writes to: opening it blocks."""
    path.parent.mkdir(parents=True, exist_ok=True)
    os.mkfifo(path)


def loop(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(path.name + ".other")
    path.with_name(path.name + ".other").symlink_to(path.name)


def device(path: Path) -> None:
    """A link to /dev/zero, which never ends: the same path inside a container."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to("/dev/zero")


# A path that breaks naive quoting: spaces, parentheses, quotes, a shell
# variable and accents.
ODD_NAME = "notes/my notes (draft)/café 'q' $HOME.txt"


def plain() -> bytes:
    return f"{MARK}\nan ordinary file\n{END}\n".encode()


# Each file with a name that says what it is, as in a real workspace, and how
# to make it: its bytes, or a function for what is not a plain file.
FILES = {
    "binary": ("build/suite.o", binary),
    "large": ("logs/suite.log", large),
    "long-line": ("dist/suite.min.js", long_line),
    "latin-1": ("notes/suite-latin1.txt", latin1),
    "missing": ("src/suite-missing.txt", None),
    "directory": ("src/suite-package", directory),
    "fifo": ("run/suite.pipe", fifo),
    "symlink-loop": ("src/suite-loop.txt", loop),
    "device": ("data/suite-zero.bin", device),
    "odd-name": (ODD_NAME, plain),
}
SPECIAL = ("directory", "fifo", "symlink-loop", "device")


def survived(run):
    """A gap unless the read got a result and the agent went on, in
    requests within the window; a skip if the read was never served."""
    if not run.served:
        return skipped("The read call was never served.")
    if any(e.kind == "over-window" for e in run.exchanges):
        return gap("over the window", "A request with the file's content was larger than the context window.")
    if not run.step_result(0, anywhere=True):
        return gap("no result", f"The read never got a result. {run.exit_detail()}".strip())
    if not run.finished_script:
        return gap("stopped", "The agent did not go on after the read.")
    return passed()


def size(text: str) -> str:
    n = len(text.encode())
    return f"{n / 1024:.0f} KB" if n < 1_000_000 else f"{n / 1048576:.1f} MB"


class Read(Scenario):
    id = "files.read"
    title = "Reading an unusual file leaves the agent working"
    description = (
        "The model reads a file with the agent's own read tool: a binary file with"
        " NUL bytes (`binary`), a 5 MB text file (`large`), a single 1 MB line"
        " (`long-line`), a file in Latin-1, not UTF-8 (`latin-1`), a path that"
        " does not exist (`missing`), a directory, a named pipe no one writes to,"
        " whose opening blocks (`fifo`), a symlink loop, or a link to `/dev/zero`,"
        " which never ends (`device`). The agent should answer with a result or an"
        " error the model can act on, keep its requests within the context window"
        " and finish the task. A plain file whose path has spaces, parentheses,"
        " quotes, `$HOME` and accents (`odd-name`) must be read. A pass says what"
        " reached the model, by the markers at the file's start and end: the"
        " whole file, a part of it, or no file content; and for `latin-1`,"
        " whether its bytes were decoded or replaced."
    )
    tools = True
    sources = ("native",)
    variants = tuple(FILES)
    timeout = 90.0
    window = 200_000
    binds = ("read",)

    def files(self):
        name, make = FILES[self.variant]
        return {name: make()} if make and self.variant not in SPECIAL else {}

    def prepare(self, sandbox):
        name, make = FILES[self.variant]
        if self.variant in SPECIAL:
            make(sandbox.host_workspace / name)

    def script(self):
        return [use(Use("read", value=FILES[self.variant][0])), DONE]

    @check(RECOMMENDED)
    def survived(self, run):
        results = run.step_result(0, anywhere=True)
        if self.variant == "odd-name" and results and MARK not in results[0].text:
            return gap("not read", f"The result: {results[0].text[:160]!r}")
        return described(survived(run), self.shown(results[0])) if results else survived(run)

    def shown(self, result):
        """What reached the model of the file, by its start and end markers."""
        text = result.text
        if MARK not in text and END not in text:
            return label("no file content", text[:160])
        if self.variant == "latin-1":
            return label("decoded" if "café" in text else "replaced", text[:120])
        whole = MARK in text and END in text
        return label("whole" if whole else "part", f"{size(text)} reached the model.")


CHANGING = "logs/suite-changing.log"
GROW_LINES, GROW_CAP = 5, 4000  # Lines per beat, and when a growing file stops.


@contextlib.contextmanager
def growing(path: Path):
    """Append numbered lines to the file every 20 ms until the context exits."""
    stop = threading.Event()

    def grow():
        n = 0
        with path.open("a") as out:
            while not stop.is_set() and n < GROW_CAP:
                out.write("".join(f"SUITE_LINE {n + i:06d} a log line\n" for i in range(GROW_LINES)))
                out.flush()
                n += GROW_LINES
                stop.wait(0.02)

    def work():
        with contextlib.suppress(OSError):  # The sandbox may go first.
            grow()

    thread = threading.Thread(target=work, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(5)


class ReadChanging(Scenario):
    id = "files.read-changing"
    title = "Reading a file that changes during the read"
    description = (
        "While the agent runs, another process appends numbered lines every 20 ms"
        " to the file the model reads, as a log grows. The agent should give a"
        " result and finish, and not wait for the file to stop growing. A pass"
        " says whether the result is a state the file had: the lines from the"
        " start, without a hole."
    )
    tools = True
    sources = ("native",)
    variants = ("growing",)
    timeout = 90.0
    window = 200_000
    binds = ("read",)

    def prepare(self, sandbox):
        path = sandbox.host_workspace / CHANGING
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("")

    def during(self, sandbox):
        return growing(sandbox.host_workspace / CHANGING)

    def script(self):
        return [use(Use("read", value=CHANGING)), DONE]

    @check(RECOMMENDED)
    def survived(self, run):
        results = run.step_result(0, anywhere=True)
        return described(survived(run), self.shown(results[0])) if results else survived(run)

    def shown(self, result):
        """Whether the result is one state of the file. Only whole lines
        count: a read can end inside a line the suite is still writing."""
        text = result.text
        numbers = [int(n) for n in re.findall(r"SUITE_LINE (\d{6}) a log line\n", text)]
        if not numbers:
            return label("empty", text[:160])
        if numbers != list(range(numbers[0], numbers[0] + len(numbers))):
            return label("lines out of order", f"{len(numbers)} lines.")
        where = "from the start" if numbers[0] == 0 else f"from line {numbers[0]}"
        return label("consistent", f"{len(numbers)} lines {where}.")


# What a write must keep: the exact bytes the model gave.
WRITES = {
    "crlf": "SUITE_WRITE\r\nwindows line ends\r\n",
    "bom": "﻿SUITE_WRITE\nafter a byte order mark\n",
    "no-final-newline": "SUITE_WRITE\nno new line at the end",
    "tabs": "SUITE_WRITE\n\tindented\twith tabs\n  and two spaces\n",
    "unicode": "SUITE_WRITE café 👩‍💻 中文 العربية\n",
    "new-folders": "SUITE_WRITE\nin folders that did not exist\n",
    "odd-name": "SUITE_WRITE\nat a path with spaces, quotes and $HOME\n",
    # 272 KB, as a generated data file: the arguments of one call stream in
    # thousands of fragments.
    "large": "SUITE_WRITE\n" + "".join(f"SUITE_LINE_{n:05d} {'abcdefghij' * 5}\n" for n in range(4000)),
}
WRITTEN = "suite-write.txt"
NESTED = "src/suite-new/deep/suite-write.txt"  # Its folders do not exist yet.


def changes(data: bytes, wanted: bytes) -> str:
    """How a written file differs from the bytes it should hold."""
    if len(data) < len(wanted) and wanted.startswith(data):
        return f"cut at {len(data)} bytes"
    found = []
    if b"\r\n" in wanted and b"\r\n" not in data:
        found.append("CRLF became LF")
    if wanted.startswith("﻿".encode()) and not data.startswith("﻿".encode()):
        found.append("BOM dropped")
    if not wanted.endswith(b"\n") and data.endswith(b"\n"):
        found.append("new line added")
    if b"\t" in wanted and b"\t" not in data:
        found.append("tabs became spaces")
    return ", ".join(found) or "changed"


class Write(Scenario):
    id = "files.write"
    title = "A write keeps the bytes it was given"
    description = (
        "The model writes a new file with the agent's own write tool: CRLF line"
        " ends, a UTF-8 byte order mark, no final new line, tabs, or Unicode; or"
        " plain text in folders that do not exist yet (`new-folders`), which the"
        " tool should create, or at a path with spaces, quotes, `$HOME` and accents"
        " (`odd-name`), or 272 KB of lines in one call (`large`). The file on disk"
        " should hold exactly the given text:"
        " a project's line ends and indentation are its own, and a changed byte"
        " shows up in every diff."
    )
    tools = True
    sources = ("native",)
    variants = tuple(WRITES)
    binds = ("write",)

    @property
    def path(self) -> str:
        return {"new-folders": NESTED, "odd-name": ODD_NAME}.get(self.variant, WRITTEN)

    def prepare(self, sandbox):
        if self.variant == "odd-name":  # Its folder exists: only the name is odd.
            (sandbox.host_workspace / ODD_NAME).parent.mkdir(parents=True)

    def script(self):
        return [use(Use("write", value=self.path, content=WRITES[self.variant])), DONE]

    def outputs(self):
        return [self.path]

    @check(RECOMMENDED)
    def exact(self, run):
        if not run.served:
            return skipped("The write call was never served.")
        data, wanted = run.outputs.get(self.path), WRITES[self.variant].encode()
        if data is None:
            results = run.step_result(0)
            return gap("not written", f"The result: {results[0].text[:160]!r}" if results else "No result came back.")
        if data == wanted:
            return passed()
        return gap(changes(data, wanted), f"The file holds {data[:80]!r}.")


GUARDED = "SUITE_GUARDED: the file as it was.\n"
NEW_TEXT = "SUITE_WRITE: the model's new text.\n"
LOCKED = "suite-readonly.txt"


class WriteGuarded(Scenario):
    id = "files.write-guarded"
    title = "What a write does to a read-only file"
    description = (
        "The model writes with the agent's own write tool to a file the user made"
        " read-only. The label says what the file on disk holds after the run:"
        " the old text (`kept`), the model's text (`overwrote`), or neither. A"
        ' read-only file often means "do not touch".'
    )
    tools = True
    sources = ("native",)
    binds = ("write",)

    def files(self):
        return {LOCKED: GUARDED.encode()}

    def prepare(self, sandbox):
        (sandbox.host_workspace / LOCKED).chmod(0o444)

    def script(self):
        return [use(Use("write", value=LOCKED, content=NEW_TEXT)), DONE]

    def outputs(self):
        return [LOCKED]

    @check(INFO)
    def wrote(self, run):
        if not run.served:
            return skipped("The write call was never served.")
        data = run.outputs.get(LOCKED)
        if data == GUARDED.encode():
            return label("kept")
        if data == NEW_TEXT.encode():
            return label("overwrote")
        return label("deleted" if data is None else "changed", "" if data is None else f"The file holds {data[:80]!r}.")


ORIGINAL = "SUITE_EDIT_START\nalpha = 1\nbeta = 2\nalpha = 1\nSUITE_EDIT_END\n"
CRLF = ORIGINAL.replace("\n", "\r\n")
EDITED = "suite-edit.txt"
# Each variant's file, and the old and new text the model gives, with "\n"
# for every new line, as models write.
EDITS = {
    "once": (ORIGINAL, "beta = 2", "beta = 3"),
    "twice": (ORIGINAL, "alpha = 1", "alpha = 9"),
    "missing": (ORIGINAL, "gamma = 3", "gamma = 4"),
    "crlf": (CRLF, "beta = 2", "beta = 3"),
    "crlf-lines": (CRLF, "alpha = 1\nbeta = 2", "alpha = 1\nbeta = 3"),
    "bom": ("\ufeff" + ORIGINAL, "beta = 2", "beta = 3"),
    "tabs": (ORIGINAL.replace("beta", "\tbeta"), "\tbeta = 2", "\tbeta = 3"),
}
REFUSED = ("twice", "missing")  # Variants whose edit should fail.


def edited(variant: str) -> str:
    """The file as a right edit leaves it: the change in place, in the
    file's own line ends."""
    original, old, new = EDITS[variant]
    if "\r\n" in original:
        old, new = old.replace("\n", "\r\n"), new.replace("\n", "\r\n")
    return original.replace(old, new)


def damage(data: str, wanted: str) -> str:
    """How an applied edit differs from the right one."""
    if "\r\n" in wanted and "\r\n" in data and data.count("\n") != data.count("\r\n"):
        return "mixed line ends"
    return changes(data.encode(), wanted.encode())


class Edit(Scenario):
    id = "files.edit"
    title = "An edit changes exactly the text it names"
    description = (
        "The model reads a file, then edits it with the agent's own edit tool. The"
        " old text occurs once (`once`), twice (`twice`), or not at all"
        " (`missing`). Once, the file must change in that place only. Twice or"
        " missing, the agent should leave the file as it was and give the model a"
        " result: a guess at the place breaks the code. The file"
        " may also have CRLF line ends, with old text of one line (`crlf`) or of"
        " two (`crlf-lines`), a UTF-8 byte order mark (`bom`), or tab indents"
        " (`tabs`). The model writes `\\n` for a new line, so the agent must find"
        " the text and keep the file's own line ends, mark and indents."
    )
    tools = True
    sources = ("native",)
    variants = tuple(EDITS)
    binds = ("read", "edit")

    def files(self):
        return {EDITED: EDITS[self.variant][0].encode()}

    def outputs(self):
        return [EDITED]

    def script(self):
        _, old, new = EDITS[self.variant]
        return [use(Use("read", value=EDITED)), use(Use("edit", value=EDITED, old=old, new=new)), DONE]

    @check(RECOMMENDED)
    def exact(self, run):
        if not run.was_served(1):
            return skipped("The edit call was never served.")
        raw = run.outputs.get(EDITED) or b""
        data = raw.decode(errors="replace")
        original, old, new = EDITS[self.variant]
        results = run.step_result(1)
        if self.variant not in REFUSED:
            wanted = edited(self.variant)
            if data == wanted:
                return passed()
            if data == original:
                said = f" The result: {results[0].text[:160]!r}" if results else ""
                return gap("not applied", f"The file is unchanged.{said}")
            return gap(damage(data, wanted), f"The file holds {raw[:120]!r}.")
        if data != original:
            replaced = data.count(new)
            return gap("replaced both" if replaced == 2 else "replaced one", f"The file holds {data[:120]!r}.")
        if not results:
            return gap("no result", "The edit got no result.")
        return passed("refused", f"The file is unchanged. The result: {results[0].text[:160]!r}")


APPENDED = "SUITE_APPENDED"  # The append command prints it once the line is in the file.


class EditStale(Scenario):
    id = "files.edit-stale"
    title = "An edit keeps a change made on disk after the read"
    description = (
        "The model reads a file, a command appends a line to it, as a formatter or"
        " the user would, and then the model edits a line the change did not touch."
        " The appended line must survive, and the edit must get a result. A pass"
        " says whether the agent applied the edit or left the file as it was."
    )
    tools = True
    sources = ("native",)
    binds = ("read", "edit")

    def files(self):
        return {EDITED: ORIGINAL.encode()}

    def outputs(self):
        return [EDITED]

    def script(self):
        return [
            use(Use("read", value=EDITED)),
            use(Use("text", command=f"printf 'SUITE_EXTERNAL\\n' >> {EDITED} && echo {APPENDED}")),
            use(Use("edit", value=EDITED, old="beta = 2", new="beta = 3")),
            DONE,
        ]

    @check(RECOMMENDED)
    def kept(self, run):
        appended = run.step_result(1)
        if not appended or APPENDED not in appended[0].text:
            return skipped("The command that appends to the file never ran.")
        if not run.was_served(2):
            return skipped("The edit was never served.")
        data = (run.outputs.get(EDITED) or b"").decode(errors="replace")
        if "SUITE_EXTERNAL" not in data:
            if "beta = 3" in data:
                return gap("change lost", "The edit wrote the file as it was read, without the appended line.")
            return gap("file changed", f"The appended line is gone, and the edit did not land: {data[:120]!r}")
        if "beta = 3" in data:
            return passed("applied")
        if not (results := run.step_result(2)):
            return gap("no result", "The edit got no result.")
        return passed("refused", f"The result: {results[0].text[:160]!r}")
