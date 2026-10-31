from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum
from typing import Any

from fastapi import HTTPException

from .config import Settings
from .metrics import Metrics
from .runtime import InferenceEngine, QueueFullError


class JobKind(StrEnum):
    DECIDE = "decide"
    RANK = "rank"
    CACHE_INFO = "cache_info"
    CACHE_CLEAR = "cache_clear"
    PRIME = "prime"


@dataclass
class Job:
    kind: JobKind
    payload: Any = None


class ModelAdapter:
    def __init__(self, model: Any):
        self.model = model

    @property
    def supports_cache(self) -> bool:
        return hasattr(self.model, "cache_info") and hasattr(self.model, "clear_cache")

    @property
    def supports_priming(self) -> bool:
        return hasattr(self.model, "prime_options")

    def decide_many(self, items: list[Any]) -> list[Any]:
        bulk = getattr(self.model, "system_one_many", None)
        if bulk is not None:
            return bulk(items)
        return [self.model.system_one(*item) for item in items]

    def rank(self, payload: Any):
        state, candidates, instructions, allow_null, top_k = payload
        return self.model.rank(
            state=state,
            candidates=candidates,
            instructions=instructions,
            allow_null=allow_null,
            top_k=top_k,
        )

    def cache_info(self):
        return self.model.cache_info()

    def clear_cache(self):
        self.model.clear_cache()
        return {"cleared": True}

    def prime(self, payload: Any):
        questions, candidates, kind = payload
        if questions is not None:
            primed = self.model.prime(questions)
        else:
            primed = self.model.prime_options(candidates, kind=kind)
        return {"primed": primed, "cache": self.model.cache_info()}


_HANDLERS: dict[JobKind, Callable[[ModelAdapter, Any], Any]] = {
    JobKind.RANK: lambda adapter, payload: adapter.rank(payload),
    JobKind.CACHE_INFO: lambda adapter, payload: adapter.cache_info(),
    JobKind.CACHE_CLEAR: lambda adapter, payload: adapter.clear_cache(),
    JobKind.PRIME: lambda adapter, payload: adapter.prime(payload),
}


def execute_jobs(model: Any, metrics: Metrics, jobs: list[Job]) -> list[Any]:
    metrics.observe_batch(len(jobs))
    adapter = ModelAdapter(model)
    results: list[Any] = [None] * len(jobs)
    decide_indices = [index for index, job in enumerate(jobs) if job.kind == JobKind.DECIDE]
    if decide_indices:
        decisions = adapter.decide_many([jobs[index].payload for index in decide_indices])
        for index, decision in zip(decide_indices, decisions, strict=True):
            results[index] = decision
    for index, job in enumerate(jobs):
        if job.kind == JobKind.DECIDE:
            continue
        handler = _HANDLERS.get(job.kind)
        if handler is None:
            raise ValueError(f"Unknown job kind: {job.kind}")
        results[index] = handler(adapter, job.payload)
    return results


async def submit(engine: InferenceEngine, settings: Settings, job: Job) -> Any:
    try:
        return await engine.submit(job, timeout=settings.request_timeout)
    except QueueFullError as exc:
        raise HTTPException(429, "Inference queue is full; retry later.") from exc
    except TimeoutError as exc:
        raise HTTPException(504, "Request timed out waiting for the model.") from exc
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(503, str(exc)) from exc
