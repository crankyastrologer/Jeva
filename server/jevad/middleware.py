from __future__ import annotations

import hmac
import time

from fastapi import FastAPI, Request

from .config import Settings
from .errors import error_response
from .metrics import Metrics
from .runtime import InferenceEngine


def install_middleware(app: FastAPI) -> None:
    @app.middleware("http")
    async def instrument(request: Request, call_next):
        settings: Settings = request.app.state.settings
        path = request.url.path
        if settings.api_key and path.startswith("/v1"):
            header = request.headers.get("authorization", "")
            if not hmac.compare_digest(header, f"Bearer {settings.api_key}"):
                return error_response(401, "Invalid or missing API key.", "invalid_api_key")
        metrics: Metrics = request.app.state.metrics
        engine: InferenceEngine | None = request.app.state.engine
        if engine is not None:
            metrics.set_queue_depth(engine.queue_depth)
        track = path.startswith("/v1")
        started = time.perf_counter()
        status = 500
        metrics.inc_inflight()
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            metrics.dec_inflight()
            if track:
                metrics.observe(path, time.perf_counter() - started)
                metrics.inc_request(path, str(status))
