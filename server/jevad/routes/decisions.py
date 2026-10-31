from __future__ import annotations

from fastapi import APIRouter

from .. import schemas as S
from ..converters import build_questions
from ..deps import EngineDep, SettingsDep
from ..execution import Job, JobKind, submit

router = APIRouter(prefix="/v1", tags=["decisions"])


@router.post("/decide", response_model=S.DecisionResponseOut)
async def decide(payload: S.DecideRequest, engine: EngineDep, settings: SettingsDep):
    questions = build_questions(payload.questions)
    response = await submit(engine, settings, Job(JobKind.DECIDE, (payload.state, questions)))
    return S.DecisionResponseOut.model_validate(response.to_dict())


@router.post("/rank", response_model=S.RankResultOut)
async def rank(payload: S.RankRequest, engine: EngineDep, settings: SettingsDep):
    job = Job(
        JobKind.RANK,
        (
            payload.state,
            payload.candidates,
            payload.instructions,
            payload.allow_null,
            payload.top_k,
        ),
    )
    result = await submit(engine, settings, job)
    return S.RankResultOut.model_validate(result.to_dict())
