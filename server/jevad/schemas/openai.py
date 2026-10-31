from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, field_validator

from .base import OpenModel


class ChatMessage(OpenModel):
    role: str
    content: str | list[dict[str, Any]] | None = None


class StreamOptions(OpenModel):
    include_usage: bool = False


class ChatCompletionRequest(OpenModel):
    model: str = "jevad"
    messages: list[ChatMessage]
    stream: bool = False
    stream_options: StreamOptions | None = None
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    user: str | None = None

    @field_validator("messages")
    @classmethod
    def _messages(cls, value: list[ChatMessage]) -> list[ChatMessage]:
        if not value:
            raise ValueError("messages must not be empty")
        return value


class CompletionRequest(OpenModel):
    model: str = "jevad"
    prompt: str | list[str]
    stream: bool = False
    stream_options: StreamOptions | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    top_p: float | None = None


class OpenAIUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ChatMessageOut(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int = 0
    message: ChatMessageOut
    finish_reason: str = "stop"
    logprobs: None = None


class ChatCompletionResponse(BaseModel):
    id: str
    object: Literal["chat.completion"] = "chat.completion"
    created: int
    model: str
    choices: list[ChatChoice]
    usage: OpenAIUsage
    system_fingerprint: str = "jeva"


class ChatDelta(BaseModel):
    role: Literal["assistant"] | None = None
    content: str | None = None


class ChatChunkChoice(BaseModel):
    index: int = 0
    delta: ChatDelta
    finish_reason: str | None = None
    logprobs: None = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatChunkChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str = "jeva"


class TextChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str | None = "stop"
    logprobs: None = None


class TextCompletionResponse(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[TextChoice]
    usage: OpenAIUsage
    system_fingerprint: str = "jeva"


class TextCompletionChunk(BaseModel):
    id: str
    object: Literal["text_completion"] = "text_completion"
    created: int
    model: str
    choices: list[TextChoice]
    usage: OpenAIUsage | None = None
    system_fingerprint: str = "jeva"
