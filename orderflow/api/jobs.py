"""In-memory background job runner for sim/exec work.

Simulations take seconds-to-a-minute and the dashboard polls for results, so
jobs run in daemon threads behind a simple registry. Results are kept in
memory only — the API is a research tool, not a service with state.
"""

from __future__ import annotations

import itertools
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass
class Job:
    id: int
    kind: str
    status: str = "queued"  # queued | running | done | error
    created_at: float = field(default_factory=time.time)
    finished_at: float | None = None
    result: dict | None = None
    error: str | None = None


_ids = itertools.count(1)
_jobs: dict[int, Job] = {}
_lock = threading.Lock()


def submit(kind: str, fn: Callable[[], dict]) -> Job:
    """Run ``fn`` in a daemon thread; its return value becomes job.result."""
    job = Job(id=next(_ids), kind=kind)
    with _lock:
        _jobs[job.id] = job

    def _run() -> None:
        job.status = "running"
        try:
            job.result = fn()
            job.status = "done"
        except Exception:  # noqa: BLE001 — surface tracebacks to the dashboard
            job.status = "error"
            job.error = traceback.format_exc(limit=8)
        finally:
            job.finished_at = time.time()

    threading.Thread(target=_run, daemon=True).start()
    return job


def get(job_id: int) -> Job | None:
    with _lock:
        return _jobs.get(job_id)


def all_jobs() -> list[Job]:
    with _lock:
        return sorted(_jobs.values(), key=lambda j: j.id, reverse=True)


def public(job: Job) -> dict:
    return {
        "id": job.id,
        "kind": job.kind,
        "status": job.status,
        "created_at": job.created_at,
        "finished_at": job.finished_at,
        "result": job.result,
        "error": job.error,
    }
