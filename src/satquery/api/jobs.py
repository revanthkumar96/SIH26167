"""In-process job store with live event fan-out.

The execution trace is streamed step by step rather than delivered at the end: a
user watching tools fire in order is watching the agentic behaviour the problem
statement asks to be made observable.

Deliberately in-process. Redis is in the architecture for a multi-worker
deployment, but a single-process store is what makes the demo runnable with one
command, and the interface here is what a Redis-backed store would implement.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

JobKind = Literal["query", "benchmark"]
JobStatus = Literal["queued", "running", "done", "error"]

#: Cap on retained jobs, so a long demo session cannot grow without bound.
MAX_JOBS = 200


@dataclass
class Job:
    """One unit of work and everything a client needs to follow it."""

    id: str
    kind: JobKind
    status: JobStatus = "queued"
    created_at: str = ""
    query: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    events: list[dict[str, Any]] = field(default_factory=list)
    result: dict[str, Any] | None = None
    error: str | None = None
    _subscribers: set[asyncio.Queue] = field(default_factory=set, repr=False)

    def summary(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "created_at": self.created_at,
            "query": self.query,
            "meta": self.meta,
            "error": self.error,
        }

    def detail(self) -> dict[str, Any]:
        return {**self.summary(), "events": self.events, "result": self.result}

    def emit(self, event: dict[str, Any]) -> None:
        """Record an event and hand it to every live subscriber.

        Never blocks and never raises: a slow or vanished websocket must not be
        able to stall the run producing the events.
        """
        self.events.append(event)
        for queue in list(self._subscribers):
            with contextlib.suppress(asyncio.QueueFull):
                queue.put_nowait(event)


class JobStore:
    """Creates jobs, streams their events, and keeps a bounded history."""

    def __init__(self, max_jobs: int = MAX_JOBS) -> None:
        self._jobs: dict[str, Job] = {}
        self._max_jobs = max_jobs

    def create(self, kind: JobKind, query: str | None = None, **meta: Any) -> Job:
        job = Job(
            id=uuid.uuid4().hex[:12],
            kind=kind,
            created_at=datetime.now(UTC).isoformat(timespec="seconds"),
            query=query,
            meta=dict(meta),
        )
        self._jobs[job.id] = job
        self._evict()
        return job

    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list(self, limit: int = 50) -> list[dict[str, Any]]:
        jobs = sorted(self._jobs.values(), key=lambda j: j.created_at, reverse=True)
        return [job.summary() for job in jobs[:limit]]

    def _evict(self) -> None:
        if len(self._jobs) <= self._max_jobs:
            return
        ordered = sorted(self._jobs.values(), key=lambda j: j.created_at)
        for job in ordered[: len(self._jobs) - self._max_jobs]:
            self._jobs.pop(job.id, None)

    async def stream(self, job: Job) -> AsyncIterator[dict[str, Any]]:
        """Replay what already happened, then follow along live.

        Replaying first closes the race where a client connects after the first
        tool has already finished and would otherwise miss it.
        """
        queue: asyncio.Queue = asyncio.Queue(maxsize=256)
        for event in list(job.events):
            queue.put_nowait(event)
        job._subscribers.add(queue)
        try:
            while True:
                event = await queue.get()
                yield event
                if event.get("type") in {"complete", "error"}:
                    return
        finally:
            job._subscribers.discard(queue)


def step_event(step: Any) -> dict[str, Any]:
    """Trace step to wire format."""
    from satquery.schema import jsonable

    return {"type": "step", "step": jsonable(step)}
