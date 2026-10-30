"""Run external callers with a wall-clock limit covering descendant processes."""
from __future__ import annotations

import os
import signal
import subprocess
import time


class MemoryLimitExceeded(subprocess.SubprocessError):
    def __init__(self, limit, observed):
        self.limit = limit
        self.observed = observed
        super().__init__(f'Observed process RSS {observed} exceeds budget {limit}')


def run_with_rss_limit(command, *, timeout, memory_bytes, stdout, stderr, poll_seconds=.1):
    """Monitor a single caller's resident memory and kill its group on a limit.

    Output must be a file rather than PIPE. The recorded budget applies to the
    direct process RSS, not virtual address space or summed descendant memory.
    Sampling permits a transient overshoot between polls.
    """
    import psutil
    if stdout == subprocess.PIPE or stderr == subprocess.PIPE:
        raise ValueError('RSS-monitored output must use files, not pipes')
    if timeout <= 0 or memory_bytes <= 0 or poll_seconds <= 0:
        raise ValueError('positive timeout, memory budget and poll interval required')
    started = time.monotonic()
    maximum = 0
    with subprocess.Popen(command, stdout=stdout, stderr=stderr, start_new_session=True) as process:
        monitored = psutil.Process(process.pid)
        try:
            while process.poll() is None:
                remaining = timeout-(time.monotonic()-started)
                if remaining <= 0:
                    raise subprocess.TimeoutExpired(command, timeout)
                try:
                    maximum = max(maximum, monitored.memory_info().rss)
                except psutil.NoSuchProcess:
                    pass
                if maximum > memory_bytes:
                    raise MemoryLimitExceeded(memory_bytes, maximum)
                try:
                    process.wait(timeout=min(poll_seconds, remaining))
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
            raise
        return subprocess.CompletedProcess(command, process.returncode), maximum


def run_with_timeout(command, *, timeout, check=False, **kwargs):
    """Match subprocess.run for caller commands, terminating the process group."""
    with subprocess.Popen(command, start_new_session=True, **kwargs) as process:
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # A wrapper such as /usr/bin/time must not leave its child running.
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.communicate()
            raise
        result = subprocess.CompletedProcess(command, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
