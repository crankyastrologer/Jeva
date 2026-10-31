from __future__ import annotations

import asyncio
import hmac
import json
import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from jeva import Choice, Noul, Score
from pydantic import ValidationError

from . import schemas as S
from .config import Settings
from .metrics import Metrics
from .runtime import InferenceEngine, QueueFullError

logger = logging.getLogger("jevad")

STATUS_TYPES = {
    400: "invalid_request_error",
    401: "authentication_error",
    403: "permission_error",
    404: "not_found_error",
    409: "conflict_error",
    422: "invalid_request_error",
    429: "rate_limit_error",
    500: "api_error",
    503: "service_unavailable",
    504: "timeout_error",
}


@dataclass
class Job:
    kind: str
    payload: Any = None


def error_response(status: int, message: str, code: str | None = None) -> JSONResponse:
    body = {
        "error": {
            "message": message,
            "type": STATUS_TYPES.get(status, "api_error"),
            "param": None,
            "code": code,
        }
    }
    return JSONResponse(status_code=status, content=body)


def build_question(spec: S.QuestionSpec):
    if isinstance(spec, S.NoulSpec):
        return Noul(instructions=spec.instructions)
    if isinstance(spec, S.ChoiceSpec):
        return Choice(
            instructions=spec.instructions,
            criteria=spec.criteria,
            allow_null=spec.allow_null,
        )
    if isinstance(spec, S.ScoreSpec):
        return Score(instructions=spec.instructions, criteria=spec.criteria, values=spec.values)
    raise ValueError(f"Unsupported question type: {type(spec).__name__}")


def build_questions(specs: S.Questions):
    if isinstance(specs, dict):
        return {name: build_question(spec) for name, spec in specs.items()}
    return [build_question(spec) for spec in specs]


def strip_fences(text: str) -> str:
    raw = text.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_decision_spec(text: str) -> S.DecideRequest:
    try:
        data = json.loads(strip_fences(text))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            400,
            "Request content must be a JSON object with 'state' and 'questions': "
            f"{exc.msg} at line {exc.lineno} column {exc.colno}",
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            400, "Request content must be a JSON object, not a JSON array or scalar."
        )
    try:
        return S.DecideRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(400, f"Invalid decision specification: {exc}") from exc


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


def chunk_text(text: str, size: int = 256) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def sse(payload: Any) -> str:
    if hasattr(payload, "model_dump_json"):
        return f"data: {payload.model_dump_json()}\n\n"
    return f"data: {json.dumps(payload)}\n\n"


def sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }


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

    @app.middleware("http")
    async def instrument(request: Request, call_next):
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
        except Exception:
            raise
        finally:
            metrics.dec_inflight()
            if track:
                metrics.observe(path, time.perf_counter() - started)
                metrics.inc_request(path, str(status))

    @app.exception_handler(RequestValidationError)
    async def on_validation(request: Request, exc: RequestValidationError):
        details = "; ".join(
            f"{'.'.join(str(part) for part in error.get('loc', ()))}: {error.get('msg')}"
            for error in exc.errors()
        )
        return error_response(400, details or "Invalid request.", "invalid_request")

    @app.exception_handler(HTTPException)
    async def on_http_error(request: Request, exc: HTTPException):
        return error_response(exc.status_code, str(exc.detail))

    @app.exception_handler(Exception)
    async def on_unhandled(request: Request, exc: Exception):
        logger.exception("Unhandled error on %s", request.url.path)
        return error_response(500, "Internal server error.")

    def engines(request: Request) -> tuple[Any, InferenceEngine]:
        engine = request.app.state.engine
        if engine is None:
            raise HTTPException(503, "Model is not loaded yet.")
        return request.app.state.model, engine

    @app.get("/")
    async def root():
        return {
            "name": "jevad",
            "model": settings.model_name,
            "docs": "/docs",
            "openai_base_url": "/v1",
        }

    @app.get("/healthz", response_model=S.HealthOut)
    async def healthz(request: Request):
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

    @app.get("/metrics", response_class=PlainTextResponse)
    async def metrics_endpoint(request: Request):
        engine = request.app.state.engine
        metrics: Metrics = request.app.state.metrics
        if engine is not None:
            metrics.set_queue_depth(engine.queue_depth)
        return PlainTextResponse(metrics.render())

    @app.get("/v1/models", response_model=S.ModelList)
    async def list_models():
        return S.ModelList(data=[S.ModelCard(id=settings.model_name, created=created)])

    @app.get("/v1/models/{model_id}", response_model=S.ModelCard)
    async def get_model(model_id: str):
        if model_id != settings.model_name:
            raise HTTPException(404, f"Model '{model_id}' not found.")
        return S.ModelCard(id=settings.model_name, created=created)

    @app.post("/v1/decide", response_model=S.DecisionResponseOut)
    async def decide(payload: S.DecideRequest, request: Request):
        active, engine = engines(request)
        questions = build_questions(payload.questions)
        response = await submit(engine, settings, Job("decide", (payload.state, questions)))
        return S.DecisionResponseOut.model_validate(response.to_dict())

    @app.post("/v1/rank", response_model=S.RankResultOut)
    async def rank(payload: S.RankRequest, request: Request):
        active, engine = engines(request)
        job = Job(
            "rank",
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

    @app.get("/v1/cache", response_model=S.CacheInfoOut)
    async def cache_info(request: Request):
        active, engine = engines(request)
        if not hasattr(active, "cache_info"):
            raise HTTPException(501, "This model does not expose cache information.")
        info = await submit(engine, settings, Job("cache_info"))
        return S.CacheInfoOut.model_validate(info)

    @app.post("/v1/cache/clear")
    async def cache_clear(request: Request):
        active, engine = engines(request)
        if not hasattr(active, "clear_cache"):
            raise HTTPException(501, "This model does not expose a cache.")
        return await submit(engine, settings, Job("cache_clear"))

    @app.post("/v1/cache/prime")
    async def cache_prime(payload: S.PrimeRequest, request: Request):
        active, engine = engines(request)
        if not hasattr(active, "prime_options"):
            raise HTTPException(501, "This model does not support cache priming.")
        questions = None
        if payload.questions is not None:
            questions = build_questions(payload.questions)
        candidates = payload.candidates
        if isinstance(candidates, dict):
            candidates = [f"{key}: {value}" for key, value in candidates.items()]
        job = Job("prime", (questions, candidates, payload.kind))
        return await submit(engine, settings, job)

    @app.post("/v1/chat/completions")
    async def chat_completions(payload: S.ChatCompletionRequest, request: Request):
        active, engine = engines(request)
        user_messages = [message for message in payload.messages if message.role == "user"]
        if not user_messages:
            raise HTTPException(400, "At least one message with role 'user' is required.")
        spec = parse_decision_spec(S.message_text(user_messages[-1]))
        questions = build_questions(spec.questions)
        decision = await submit(engine, settings, Job("decide", (spec.state, questions)))
        content = json.dumps(decision.to_dict(), ensure_ascii=False)
        usage = S.openai_usage(decision.usage)
        completion_id = S.new_id("chatcmpl")
        stamp = S.now()

        if not payload.stream:
            return S.ChatCompletionResponse(
                id=completion_id,
                created=stamp,
                model=settings.model_name,
                choices=[S.ChatChoice(message=S.ChatMessageOut(content=content))],
                usage=usage,
            )

        include_usage = bool(payload.stream_options and payload.stream_options.include_usage)

        async def events():
            yield sse(
                S.ChatCompletionChunk(
                    id=completion_id,
                    created=stamp,
                    model=settings.model_name,
                    choices=[S.ChatChunkChoice(delta=S.ChatDelta(role="assistant", content=""))],
                )
            )
            for piece in chunk_text(content):
                yield sse(
                    S.ChatCompletionChunk(
                        id=completion_id,
                        created=stamp,
                        model=settings.model_name,
                        choices=[S.ChatChunkChoice(delta=S.ChatDelta(content=piece))],
                    )
                )
            yield sse(
                S.ChatCompletionChunk(
                    id=completion_id,
                    created=stamp,
                    model=settings.model_name,
                    choices=[S.ChatChunkChoice(delta=S.ChatDelta(), finish_reason="stop")],
                )
            )
            if include_usage:
                yield sse(
                    S.ChatCompletionChunk(
                        id=completion_id,
                        created=stamp,
                        model=settings.model_name,
                        choices=[],
                        usage=usage,
                    )
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers=sse_headers())

    @app.post("/v1/completions")
    async def completions(payload: S.CompletionRequest, request: Request):
        active, engine = engines(request)
        prompt = payload.prompt[0] if isinstance(payload.prompt, list) else payload.prompt
        if not isinstance(prompt, str) or not prompt.strip():
            raise HTTPException(400, "prompt must be a nonempty string.")
        spec = parse_decision_spec(prompt)
        questions = build_questions(spec.questions)
        decision = await submit(engine, settings, Job("decide", (spec.state, questions)))
        text = json.dumps(decision.to_dict(), ensure_ascii=False)
        usage = S.openai_usage(decision.usage)
        completion_id = S.new_id("cmpl")
        stamp = S.now()

        if not payload.stream:
            return S.TextCompletionResponse(
                id=completion_id,
                created=stamp,
                model=settings.model_name,
                choices=[S.TextChoice(text=text)],
                usage=usage,
            )

        include_usage = bool(payload.stream_options and payload.stream_options.include_usage)

        async def events():
            for piece in chunk_text(text):
                yield sse(
                    S.TextCompletionChunk(
                        id=completion_id,
                        created=stamp,
                        model=settings.model_name,
                        choices=[S.TextChoice(text=piece, finish_reason=None)],
                    )
                )
            yield sse(
                S.TextCompletionChunk(
                    id=completion_id,
                    created=stamp,
                    model=settings.model_name,
                    choices=[S.TextChoice(text="", finish_reason="stop")],
                )
            )
            if include_usage:
                yield sse(
                    S.TextCompletionChunk(
                        id=completion_id,
                        created=stamp,
                        model=settings.model_name,
                        choices=[],
                        usage=usage,
                    )
                )
            yield "data: [DONE]\n\n"

        return StreamingResponse(events(), media_type="text/event-stream", headers=sse_headers())

    return app


app = create_app()
