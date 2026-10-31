from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

from . import schemas as S
from .openai_spec import openai_usage
from .streaming import chunk_text, sse


def decision_payload(decision: Any) -> tuple[str, S.OpenAIUsage]:
    return json.dumps(decision.to_dict(), ensure_ascii=False), openai_usage(decision.usage)


def chat_response(content: str, usage: S.OpenAIUsage, model_name: str) -> S.ChatCompletionResponse:
    return S.ChatCompletionResponse(
        id=S.new_id("chatcmpl"),
        created=S.now(),
        model=model_name,
        choices=[S.ChatChoice(message=S.ChatMessageOut(content=content))],
        usage=usage,
    )


def chat_stream(
    content: str,
    usage: S.OpenAIUsage,
    model_name: str,
    include_usage: bool,
) -> AsyncIterator[str]:
    completion_id = S.new_id("chatcmpl")
    stamp = S.now()

    def chunk(delta: S.ChatDelta, finish_reason: str | None = None) -> S.ChatCompletionChunk:
        return S.ChatCompletionChunk(
            id=completion_id,
            created=stamp,
            model=model_name,
            choices=[S.ChatChunkChoice(delta=delta, finish_reason=finish_reason)],
        )

    async def events() -> AsyncIterator[str]:
        yield sse(chunk(S.ChatDelta(role="assistant", content="")))
        for piece in chunk_text(content):
            yield sse(chunk(S.ChatDelta(content=piece)))
        yield sse(chunk(S.ChatDelta(), "stop"))
        if include_usage:
            yield sse(
                S.ChatCompletionChunk(
                    id=completion_id,
                    created=stamp,
                    model=model_name,
                    choices=[],
                    usage=usage,
                )
            )
        yield "data: [DONE]\n\n"

    return events()


def text_response(
    content: str, usage: S.OpenAIUsage, model_name: str
) -> S.TextCompletionResponse:
    return S.TextCompletionResponse(
        id=S.new_id("cmpl"),
        created=S.now(),
        model=model_name,
        choices=[S.TextChoice(text=content)],
        usage=usage,
    )


def text_stream(
    content: str,
    usage: S.OpenAIUsage,
    model_name: str,
    include_usage: bool,
) -> AsyncIterator[str]:
    completion_id = S.new_id("cmpl")
    stamp = S.now()

    def chunk(text: str, finish_reason: str | None = None) -> S.TextCompletionChunk:
        return S.TextCompletionChunk(
            id=completion_id,
            created=stamp,
            model=model_name,
            choices=[S.TextChoice(text=text, finish_reason=finish_reason)],
        )

    async def events() -> AsyncIterator[str]:
        for piece in chunk_text(content):
            yield sse(chunk(piece))
        yield sse(chunk("", "stop"))
        if include_usage:
            yield sse(
                S.TextCompletionChunk(
                    id=completion_id,
                    created=stamp,
                    model=model_name,
                    choices=[],
                    usage=usage,
                )
            )
        yield "data: [DONE]\n\n"

    return events()
