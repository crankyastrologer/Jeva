from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from .. import schemas as S
from ..converters import build_questions
from ..deps import EngineDep, SettingsDep
from ..execution import Job, JobKind, submit
from ..openai_responses import (
    chat_response,
    chat_stream,
    decision_payload,
    text_response,
    text_stream,
)
from ..openai_spec import message_text, parse_decision_spec
from ..streaming import sse_headers

router = APIRouter(prefix="/v1", tags=["openai"])


async def _decide(spec: S.DecideRequest, engine: EngineDep, settings: SettingsDep):
    questions = build_questions(spec.questions)
    return await submit(engine, settings, Job(JobKind.DECIDE, (spec.state, questions)))


@router.post("/chat/completions")
async def chat_completions(
    payload: S.ChatCompletionRequest,
    engine: EngineDep,
    settings: SettingsDep,
):
    user_messages = [message for message in payload.messages if message.role == "user"]
    if not user_messages:
        raise HTTPException(400, "At least one message with role 'user' is required.")
    decision = await _decide(parse_decision_spec(message_text(user_messages[-1])), engine, settings)
    content, usage = decision_payload(decision)
    model_name = settings.model_name

    if not payload.stream:
        return chat_response(content, usage, model_name)
    include_usage = bool(payload.stream_options and payload.stream_options.include_usage)
    events = chat_stream(content, usage, model_name, include_usage)
    return StreamingResponse(events, media_type="text/event-stream", headers=sse_headers())


@router.post("/completions")
async def completions(payload: S.CompletionRequest, engine: EngineDep, settings: SettingsDep):
    prompt = payload.prompt[0] if isinstance(payload.prompt, list) else payload.prompt
    if not isinstance(prompt, str) or not prompt.strip():
        raise HTTPException(400, "prompt must be a nonempty string.")
    decision = await _decide(parse_decision_spec(prompt), engine, settings)
    content, usage = decision_payload(decision)
    model_name = settings.model_name

    if not payload.stream:
        return text_response(content, usage, model_name)
    include_usage = bool(payload.stream_options and payload.stream_options.include_usage)
    events = text_stream(content, usage, model_name, include_usage)
    return StreamingResponse(events, media_type="text/event-stream", headers=sse_headers())
