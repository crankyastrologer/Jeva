from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import Field, field_validator, model_validator

from .base import OpenModel


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
