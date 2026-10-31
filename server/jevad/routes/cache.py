from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import schemas as S
from ..converters import build_questions
from ..deps import EngineDep, ModelDep, SettingsDep
from ..execution import Job, submit

router = APIRouter(prefix="/v1/cache", tags=["cache"])


@router.get("", response_model=S.CacheInfoOut)
async def cache_info(model: ModelDep, engine: EngineDep, settings: SettingsDep):
    if not hasattr(model, "cache_info"):
        raise HTTPException(501, "This model does not expose cache information.")
    info = await submit(engine, settings, Job("cache_info"))
    return S.CacheInfoOut.model_validate(info)


@router.post("/clear")
async def cache_clear(model: ModelDep, engine: EngineDep, settings: SettingsDep):
    if not hasattr(model, "clear_cache"):
        raise HTTPException(501, "This model does not expose a cache.")
    return await submit(engine, settings, Job("cache_clear"))


@router.post("/prime")
async def cache_prime(
    payload: S.PrimeRequest,
    model: ModelDep,
    engine: EngineDep,
    settings: SettingsDep,
):
    if not hasattr(model, "prime_options"):
        raise HTTPException(501, "This model does not support cache priming.")
    questions = None
    if payload.questions is not None:
        questions = build_questions(payload.questions)
    candidates = payload.candidates
    if isinstance(candidates, dict):
        candidates = [f"{key}: {value}" for key, value in candidates.items()]
    job = Job("prime", (questions, candidates, payload.kind))
    return await submit(engine, settings, job)
