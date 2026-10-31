from __future__ import annotations

from typing import Any

from fastapi import FastAPI

from . import schemas as S
from .config import Settings
from .errors import install_error_handlers
from .lifecycle import build_lifespan
from .metrics import Metrics
from .middleware import install_middleware
from .routes import cache, decisions, health, models, openai


def create_app(settings: Settings | None = None, model: Any | None = None) -> FastAPI:
    settings = settings or Settings.from_env()
    app = FastAPI(
        title="jevad",
        version="0.2.0",
        description="OpenAI-compatible inference server for Jeva typed decisions.",
        lifespan=build_lifespan(settings),
    )
    app.state.settings = settings
    app.state.model_override = model
    app.state.model = model
    app.state.engine = None
    app.state.metrics = Metrics()
    app.state.created = S.now()

    install_middleware(app)
    install_error_handlers(app)
    app.include_router(health.router)
    app.include_router(models.router)
    app.include_router(decisions.router)
    app.include_router(cache.router)
    app.include_router(openai.router)
    return app


app = create_app()
