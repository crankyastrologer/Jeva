from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from fastapi import HTTPException

from .config import Settings
from .metrics import Metrics
from .runtime import InferenceEngine, QueueFullError


@dataclass
class Job:
    kind: str
    payload: Any = None


def execute_jobs(model: Any, metrics: Metrics, jobs: list[Job]) -> list[Any]:
    metrics.observe_batch(len(jobs))
    results: list[Any] = [None] * len(jobs)
    decide_indices = [index for index, job in enumerate(jobs) if job.kind == "decide"]
    if decide_indices:
        items = [jobs[index].payload for index in decide_indices]
        batched = getattr(model, "system_one_many", None)
        decisions = batched(items) if batched else [model.system_one(*item) for item in items]
        for index, decision in zip(decide_indices, decisions, strict=True):
            results[index] = decision
    for index, job in enumerate(jobs):
        if job.kind == "rank":
            state, candidates, instructions, allow_null, top_k = job.payload
            results[index] = model.rank(
                state=state,
                candidates=candidates,
                instructions=instructions,
                allow_null=allow_null,
                top_k=top_k,
            )
        elif job.kind == "cache_info":
            results[index] = model.cache_info()
        elif job.kind == "cache_clear":
            model.clear_cache()
            results[index] = {"cleared": True}
        elif job.kind == "prime":
            questions, candidates, kind = job.payload
            if questions is not None:
                primed = model.prime(questions)
            else:
                primed = model.prime_options(candidates, kind=kind)
            results[index] = {"primed": primed, "cache": model.cache_info()}
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
