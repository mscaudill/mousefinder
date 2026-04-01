"""Tools for measuring or assigning memory, compute or file resources."""

import os
import platform
from datetime import datetime
from pathlib import Path

import psutil
from statx import statx


def allocate(
    jobs: int,
    requesting: int | None = None,
    hyperthread: bool = False,
) -> int:
    """Allocates requested number of cores to run jobs.

    Args:
        jobs:
            The number of tasks to be executed in parallel.
        requesting:
            The number of cores being requested to operate on jobs in
            parallel. If None, the cores assigned will be the maximum available
            on the system.
        hyperthread:
            A boolean indicating if hyperthreaded cores should assign individual
            threads (True) or only physical cores (False). For CPU bound task,
            hyperthreading will not help and should be set to False. For io
            dependent tasks (i.e. short compute times) hyperthreading can
            lead to dramatic speed-ups.

    Returns:
        The minimum of jobs, requesting and available cores. If the number of
        threads or CPUs can not be determined allocate will return a single
        core.
    """

    requested = jobs if requesting is None else requesting
    # get hyperthread count to get available physical cores
    threads, cpus = psutil.cpu_count(), psutil.cpu_count(False)
    if threads and cpus:

        threads_per_core = threads // cpus
        available = len(psutil.Process().cpu_affinity())  # type: ignore
        if not hyperthread:
            available //= threads_per_core

        return min(jobs, requested, available)

    return 1
