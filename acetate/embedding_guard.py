"""Cross-process audio-compute locks, shared by the sidecar and local workers."""

from __future__ import annotations

import fcntl
from contextlib import contextmanager
from pathlib import Path

BUSY_EXIT = 75


@contextmanager
def slot(data: Path, name: str = "inference"):
    """Try once, never wait or unlink the lock file. The OS releases it on exit.

    All containers must mount the same data directory. Locking the shared file
    also covers independent CLI invocations and CLAP text queries on that host.
    """
    data.mkdir(parents=True, exist_ok=True)
    with (data / f".embedding-{name}.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            yield False
            return
        try:
            yield True
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def quiescent(data: Path):
    """Keep workers out for the full duration of an offline database swap."""
    with slot(data, "requests") as controller, slot(data) as inference:
        if not controller or not inference:
            raise RuntimeError(
                "Stop the embedding controller and inference workers before replacing the database"
            )
        yield
