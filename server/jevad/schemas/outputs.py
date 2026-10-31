from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field


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


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


def now() -> int:
    return int(time.time())
