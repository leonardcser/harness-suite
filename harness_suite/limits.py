"""Memory limits, so that an agent that runs away cannot take the machine
down with it.

Everything a run starts lives in one systemd user slice with a cap on the
total: the suite itself, and every agent. Each agent also has its own cap. An
agent that reaches its cap is killed alone, and its case records the kill as
a finding. Without a systemd user manager, the suite runs without limits and
says so.
"""

import functools
import os
import subprocess
import sys
import time
from pathlib import Path

# A dash in a slice name nests it, so the name has none.
SLICE = "harnesssuite.slice"
AGENT_MEMORY = 4 * 1024**3  # Bytes one agent may use, with its tools.
TOTAL_SHARE = 0.75  # Of the machine's memory, for everything in the slice.
INSIDE = "HARNESS_SUITE_SCOPE"  # Set once the suite runs in its scope.


def total_memory() -> int:
    return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")


@functools.cache
def available() -> bool:
    """Whether a systemd user manager can make capped scopes."""
    try:
        probe = subprocess.run(
            ["systemd-run", "--user", "--scope", "--quiet", "-p", "MemoryMax=64M", "true"],
            capture_output=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return probe.returncode == 0


def scope(memory: int, unit: str | None = None) -> list[str]:
    """The words that start a command in its own capped scope in the slice,
    or none without systemd. The command keeps its PID and environment."""
    if not available():
        return []
    return [
        "systemd-run",
        "--user",
        "--scope",
        "--quiet",
        "--collect",
        *([f"--unit={unit}"] if unit else []),
        f"--slice={SLICE}",
        "-p",
        f"MemoryMax={memory}",
        "-p",
        "MemorySwapMax=0",
        "--",
    ]


def release(unit: str) -> None:
    """End a scope once its agent ended: every process left in it is killed
    at once, and the scope stops without waiting. systemd stops an empty
    scope by itself only while it can watch the scope's cgroup."""
    if available():
        name = f"{unit}.scope"
        subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", name], capture_output=True)
        subprocess.run(["systemctl", "--user", "stop", "--no-block", name], capture_output=True)


def container() -> list[str]:
    """The podman options that cap a container and put it in the slice."""
    if not available():
        return []
    memory = f"--memory={AGENT_MEMORY}"
    return [memory, f"--memory-swap={AGENT_MEMORY}", f"--cgroup-parent={SLICE}"]


def enter() -> None:
    """Run the rest of this process in a scope in the slice, with the slice
    capped at a share of the machine's memory. Returns at once inside it."""
    if os.environ.get(INSIDE):
        return
    if not available():
        print("harness-suite: no systemd user manager; running without memory limits", file=sys.stderr)
        return
    total = int(total_memory() * TOTAL_SHARE)
    subprocess.run(
        ["systemctl", "--user", "set-property", "--runtime", SLICE, f"MemoryMax={total}", "MemorySwapMax=0"],
        check=True,
        capture_output=True,
    )
    os.environ[INSIDE] = "1"
    argv = [sys.executable, "-m", "harness_suite", *sys.argv[1:]]
    os.execvp("systemd-run", [*scope(total), *argv])


TYPICAL_AGENT = 1024**3  # Bytes a running agent uses, with room to spare: most use half.
MEMORY_START = 0.6  # Share of the slice's cap that agents may use when a case starts.
CPU_STALL = 0.3  # Share of time some task waits for a core, above which no case starts.
CPU_SAMPLE = 0.25  # Seconds over which `wait_for_room` measures.
JOBS_PER_CORE = 2


def jobs() -> int:
    """The most cases to run at once. Most cases wait on an agent, so this is
    a ceiling, and `wait_for_room` decides when the next one starts. Without
    the slice to watch, it also keeps to a typical agent's share of memory."""
    most = JOBS_PER_CORE * (os.cpu_count() or 1)
    if not available():
        most = min(most, int(total_memory() * TOTAL_SHARE) // TYPICAL_AGENT)
    return max(1, most)


def cpu_stalled() -> int | None:
    """Microseconds since boot that some task waited for a core (Linux
    pressure stall information), or None where the kernel does not say."""
    try:
        with open("/proc/pressure/cpu") as pressure:
            return int(pressure.readline().rsplit("total=", 1)[1])
    except (OSError, IndexError, ValueError):
        return None


_slice_folder: Path | None = None


def slice_folder() -> Path | None:
    """The slice's cgroup folder, or None while systemd has none for it."""
    global _slice_folder
    if _slice_folder is None and available():
        shown = subprocess.run(
            ["systemctl", "--user", "show", "-p", "ControlGroup", "--value", SLICE], capture_output=True, text=True
        ).stdout.strip()
        if shown and (folder := Path("/sys/fs/cgroup", shown.lstrip("/"))).is_dir():
            _slice_folder = folder
    return _slice_folder


def held(folder: Path) -> int:
    """Bytes the processes of a cgroup hold. Page cache is left out: the
    kernel takes it back first."""
    stat = dict(line.split() for line in (folder / "memory.stat").read_text().splitlines())
    return int(stat["anon"]) + int(stat["shmem"])


def slice_memory() -> tuple[int, int] | None:
    """(bytes the slice's processes hold, the slice's cap), or None outside
    a capped slice."""
    folder = slice_folder()
    try:
        return (held(folder), int((folder / "memory.max").read_text())) if folder else None
    except (OSError, KeyError, ValueError):
        return None


def memory_full() -> bool:
    """Whether another agent would take the slice past `MEMORY_START` of its cap."""
    memory = slice_memory()
    return bool(memory) and memory[0] + TYPICAL_AGENT > MEMORY_START * memory[1]


def stall_share() -> float | None:
    """The share of the next `CPU_SAMPLE` seconds in which some task waited
    for a core, or None where the kernel does not say."""
    before = cpu_stalled()
    time.sleep(CPU_SAMPLE)
    after = cpu_stalled()
    return None if before is None or after is None else (after - before) / 1e6 / CPU_SAMPLE


def wait_for_room() -> None:
    """Wait until another agent fits: in the slice's memory, and on the cores.
    Busy cores are fine; an agent that waits for one can miss input and time
    out. Where the kernel does not say, there is room."""
    while True:
        if memory_full():
            time.sleep(CPU_SAMPLE)
            continue
        share = stall_share()
        if share is None or share < CPU_STALL:
            return
