from __future__ import annotations

import time
import uuid
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class OpenModel(BaseModel):
    model_config = ConfigDict(extra="allow")


class NoulSpec(OpenModel):
    type: Literal["noul"] = "noul"
    instructions: str


class ChoiceSpec(OpenModel):
    type: Literal["choice"] = "choice"
    instructions: str
    criteria: dict[str, str] | list[str]
    allow_null: bool = False


class ScoreSpec(OpenModel):
    type: Literal["score"] = "score"
    instructions: str
    criteria: dict[str, str] | list[str]
    values: list[float] | None = None


QuestionSpec = Annotated[NoulSpec | ChoiceSpec | ScoreSpec, Field(discriminator="type")]
Questions = dict[str, QuestionSpec] | list[QuestionSpec]


class DecideRequest(OpenModel):
    state: str | dict[str, Any]
    questions: Questions

    @field_validator("questions")
    @classmethod
    def _nonempty(cls, value: Questions) -> Questions:
        if not value:
            raise ValueError("questions must not be empty")
        return value


class RankRequest(OpenModel):
    state: str | dict[str, Any]
    candidates: dict[str, str] | list[str]
    instructions: str = "Rank candidates by relevance to the state."
    allow_null: bool = False
    top_k: int | None = None

    @field_validator("candidates")
    @classmethod
    def _candidates(cls, value: dict[str, str] | list[str]) -> dict[str, str] | list[str]:
        if not value:
            raise ValueError("candidates must not be empty")
        return value

    @field_validator("top_k")
    @classmethod
    def _top_k(cls, value: int | None) -> int | None:
        if value is not None and value < 1:
            raise ValueError("top_k must be a positive integer")
        return value


class PrimeRequest(OpenModel):
    questions: Questions | None = None
    candidates: dict[str, str] | list[str] | None = None
    kind: Literal["choice", "score", "noul"] = "choice"

    @model_validator(mode="after")
    def _single_source(self):
        if (self.questions is None) == (self.candidates is None):
            raise ValueError("provide exactly one of 'questions' or 'candidates'")
        if self.candidates is not None and not self.candidates:
            raise ValueError("candidates must not be empty")
        return self


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


class UsageOut(BaseModel):
    input_tokens: int = 0
    requested_input_tokens: int = 0
    encoded_state_tokens: int = 0
    encoded_option_tokens: int = 0
    cached_option_tokens: int = 0
    output_tokens: int = 0
    truncated_tokens: int = 0


class AnswerOut(BaseModel):
    kind: str
    probabilities: dict[str, float]
    conditional_probabilities: dict[str, float]
    null_probability: float
    choice: str | None = None
    noul: float | None = None
    score: float | None = None
    confidence: float = 0.0
    candidate_probabilities: tuple[float, ...] = ()
    usage: UsageOut = UsageOut()
    latency_ms: float = 0.0


class DecisionResponseOut(BaseModel):
    answers: dict[str, AnswerOut] | list[AnswerOut]
    usage: UsageOut
    latency_ms: float


class RankedItemOut(BaseModel):
    label: str
    probability: float
    index: int


class RankResultOut(BaseModel):
    ranking: list[RankedItemOut]
    null_probability: float
    usage: UsageOut
    latency_ms: float


class CacheInfoOut(BaseModel):
    mode: str
    options: dict[str, Any]
    sets: dict[str, Any]
    states: dict[str, Any] = Field(default_factory=dict)


class HealthOut(BaseModel):
    status: str
    model: str
    device: str
    cache: str
    queue: dict[str, int]


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str = "jeva"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


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


class ErrorDetail(BaseModel):
    message: str
    type: str
    param: str | None = None
    code: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorDetail


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def now() -> int:
    return int(time.time())
