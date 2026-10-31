from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from .config import Settings
from .execution import execute_jobs
from .metrics import Metrics
from .runtime import InferenceEngine

logger = logging.getLogger("jevad")


def load_model(settings: Settings) -> Any:
    from jeva import Jeva

    logger.info("Loading Jeva checkpoint %s on %s", settings.model, settings.device)
    model = Jeva.from_pretrained(
        settings.model,
        device=settings.device,
        cache=settings.cache,
        option_chunk_size=settings.option_chunk_size,
        option_batch_tokens=settings.option_batch_tokens,
        max_state_length=settings.max_state_length,
        max_option_length=settings.max_option_length,
        local_files_only=settings.local_files_only,
        quantize=settings.quantize,
        compile_core=settings.compile_core,
    )
    logger.info("Jeva checkpoint loaded")
    return model


def build_engine(model: Any, metrics: Metrics, settings: Settings) -> InferenceEngine:
    return InferenceEngine(
        lambda jobs: execute_jobs(model, metrics, jobs),
        queue_size=settings.queue_size,
        default_timeout=settings.request_timeout,
        batch_window=settings.batch_window_ms / 1000.0,
        max_batch=settings.max_batch,
    )


def build_lifespan(settings: Settings):
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = app.state.model_override
        if active is None:
            active = await asyncio.to_thread(load_model, settings)
        metrics: Metrics = app.state.metrics
        engine = build_engine(active, metrics, settings)
        await engine.start()
        app.state.model = active
        app.state.engine = engine
        try:
            yield
        finally:
            await engine.stop()

    return lifespan
