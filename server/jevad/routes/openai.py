from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import schemas as S
from ..converters import build_questions
from ..deps import EngineDep, SettingsDep
from ..execution import Job, submit
from ..openai_spec import message_text, openai_usage, parse_decision_spec
from ..streaming import chunk_text, sse, sse_headers

router = APIRouter(prefix="/v1", tags=["openai"])


@router.post("/chat/completions")
async def chat_completions(
    payload: S.ChatCompletionRequest,
    engine: EngineDep,
    settings: SettingsDep,
):
    user_messages = [message for message in payload.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(400, "At least one message with role 'user' is required.")
    spec = parse_decision_spec(message_text(user_messages[-1]))
    questions = build_questions(spec.questions)
    decision = await submit(engine, settings, Job("decide", (spec.state, questions)))
    content = json.dumps(decision.to_dict(), ensure_ascii=False)
    usage = openai_usage(decision.usage)
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


@router.post("/completions")
async def completions(payload: S.CompletionRequest, engine: EngineDep, settings: SettingsDep):
    prompt = payload.prompt[0] if isinstance(payload.prompt, list) else payload.prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "prompt must be a nonempty string.")
    spec = parse_decision_spec(prompt)
    questions = build_questions(spec.questions)
    decision = await submit(engine, settings, Job("decide", (spec.state, questions)))
    text = json.dumps(decision.to_dict(), ensure_ascii=False)
    usage = openai_usage(decision.usage)
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
