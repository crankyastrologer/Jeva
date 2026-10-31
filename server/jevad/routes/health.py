from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import PlainTextResponse

from .. import schemas as S
from ..deps import SettingsDep
from ..metrics import Metrics

router = APIRouter()


@router.get("/")
async def root(settings: SettingsDep):
    return {
        "name": "jevad",
        "model": settings.model_name,
        "docs": "/docs",
        "openai_base_url": "/v1",
    }


@router.get("/healthz", response_model=S.HealthOut)
async def healthz(request: Request, settings: SettingsDep):
    engine = request.app.state.engine
    model_loaded = request.app.state.model is not None and engine is not None
    return S.HealthOut(
        status="ok" if model_loaded else "starting",
        model=settings.model_name,
        device=settings.device,
        cache=settings.cache,
        queue={
            "depth": 0 if engine is None else engine.queue_depth,
            "capacity": 0 if engine is None else engine.queue_size,
        },
    )


@router.get("/metrics", response_class=PlainTextResponse)
async def metrics_endpoint(request: Request):
    engine = request.app.state.engine
    metrics: Metrics = request.app.state.metrics
    if engine is not None:
        metrics.set_queue_depth(engine.queue_depth)
    return PlainTextResponse(metrics.render())
