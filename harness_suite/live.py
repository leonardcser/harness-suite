"""Every agent process the suite has started and not yet stopped, so that an
error or Ctrl-C leaves no container, process or tmux server behind."""

import threading
import uuid

_lock = threading.Lock()
_running: set = set()
_closed = False  # After `stop_all`, anything new is stopped at once.


def token() -> str:
    return uuid.uuid4().hex[:12]


def add(process) -> None:
    with _lock:
        _running.add(process)
        closed = _closed
    if closed:
        process.stop()


def remove(process) -> None:
    with _lock:
        _running.discard(process)


def stop_all() -> None:
    """Stop everything still running, and all that starts later. Each item has `stop()`."""
    global _closed
    with _lock:
        _closed = True
        running = list(_running)
    for process in running:
        try:
            process.stop()
        except Exception:
            pass
