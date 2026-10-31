from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI

from . import schemas as S
from .config import Settings
from .errors import install_error_handlers
from .execution import execute_jobs
from .metrics import Metrics
from .middleware import install_middleware
from .routes import cache, decisions, health, models, openai
from .runtime import InferenceEngine

logger = logging.getLogger("jevad")


def create_app(settings: Settings | None = None, model: Any | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    created = S.now()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        active = app.state.model_override
        if active is None:
            from jeva import Jeva

            logger.info("Loading Jeva checkpoint %s on %s", settings.model, settings.device)
            active = await asyncio.to_thread(
                Jeva.from_pretrained,
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
        metrics: Metrics = app.state.metrics
        engine = InferenceEngine(
            lambda jobs: execute_jobs(active, metrics, jobs),
            queue_size=settings.queue_size,
            default_timeout=settings.request_timeout,
            batch_window=settings.batch_window_ms / 1000.0,
            max_batch=settings.max_batch,
        )
        await engine.start()
        app.state.model = active
        app.state.engine = engine
        try:
            yield
        finally:
            await engine.stop()

    app = FastAPI(
        title="jevad",
        version="0.2.0",
        description="OpenAI-compatible inference server for Jeva typed decisions.",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.model_override = model
    app.state.model = model
    app.state.engine = None
    app.state.metrics = Metrics()
    app.state.created = created

    install_middleware(app)
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(decisions.router)
    app.include_router(cache.router)
    app.include_router(openai.router)
    return app


app = create_app()
