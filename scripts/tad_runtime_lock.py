"""Serialize local TAD callers to avoid competing high-memory timing runs."""
from contextlib import contextmanager
import fcntl
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]


@contextmanager
def caller_lock():
    path=ROOT/'artifacts/tad_runtime_comparison_20261024/caller.lock'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a+') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle,fcntl.LOCK_UN)
