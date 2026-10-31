from __future__ import annotations

from jeva import Choice, Noul, Score
from jeva.types import Answer, DecisionResponse, RankedItem, RankResult, Usage

from .app import create_app
from .config import Settings


class StubJeva:
    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.batch_sizes: list[int] = []

    def _usage(self) -> Usage:
        return Usage(
            input_tokens=12,
            requested_input_tokens=12,
            encoded_state_tokens=8,
            encoded_option_tokens=4,
            cached_option_tokens=0,
            output_tokens=0,
        )

    @staticmethod
    def _labels(question) -> list[str]:
        if isinstance(question, Noul):
            return ["false", "true"]
        criteria = question.criteria
        return list(criteria) if isinstance(criteria, dict) else list(criteria)

    def _answer(self, question) -> Answer:
        labels = self._labels(question)
        if isinstance(question, Noul):
            return Answer(
                kind="noul",
                probabilities={"false": 0.2, "true": 0.8},
                conditional_probabilities={"false": 0.2, "true": 0.8},
                null_probability=0.0,
                noul=0.8,
                confidence=0.8,
            )
        probabilities = {label: 1.0 / len(labels) for label in labels}
        if isinstance(question, Choice):
            return Answer(
                kind="choice",
                probabilities=probabilities,
                conditional_probabilities=probabilities,
                null_probability=0.0,
                choice=labels[0],
                confidence=1.0 / len(labels),
            )
        assert isinstance(question, Score)
        values = question.values or list(range(len(labels)))
        score = sum(p * v for p, v in zip(probabilities.values(), values, strict=True))
        return Answer(
            kind="score",
            probabilities=probabilities,
            conditional_probabilities=probabilities,
            null_probability=0.0,
            score=score,
            confidence=1.0 / len(labels),
        )

    def system_one(self, state, questions):
        return self._response(state, questions)

    def system_one_many(self, requests):
        self.batch_sizes.append(len(requests))
        return [self._response(state, questions) for state, questions in requests]

    def _response(self, state, questions):
        self.calls.append(("system_one", state, questions))
        named = isinstance(questions, dict)
        items = list(questions.items()) if named else list(enumerate(questions))
        answers = {}
        for name, question in items:
            answer = self._answer(question)
            answer.usage = self._usage()
            answer.latency_ms = 1.5
            answers[name] = answer
        return DecisionResponse(
            answers=answers if named else list(answers.values()),
            usage=self._usage(),
            latency_ms=2.0,
        )

    def rank(self, state, candidates, *, instructions="", allow_null=False, top_k=None):
        self.calls.append(("rank", state, candidates))
        labels = list(candidates)
        ranking = [
            RankedItem(label=label, probability=1.0 / len(labels), index=index)
            for index, label in enumerate(labels)
        ]
        if top_k is not None:
            ranking = ranking[:top_k]
        return RankResult(
            ranking=ranking, null_probability=0.0, usage=self._usage(), latency_ms=3.0
        )

    def cache_info(self):
        return {
            "mode": "hybrid",
            "options": {"entries": 0, "bytes": 0},
            "sets": {"entries": 0, "bytes": 0},
            "states": {"entries": 0, "bytes": 0},
        }

    def clear_cache(self):
        self.calls.append(("clear_cache",))

    def prime_options(self, options, *, kind="choice"):
        self.calls.append(("prime_options", kind, list(options)))
        return {
            "kind": kind,
            "options": len(list(options)),
            "encoded_options": len(list(options)),
            "encoded_tokens": 0,
            "cached_tokens": 0,
            "truncated_tokens": 0,
            "set_cached": True,
            "latency_ms": 0.5,
        }

    def prime_question(self, question):
        labels = self._labels(question)
        return self.prime_options(labels, kind="choice")

    def prime(self, questions):
        items = questions.items() if isinstance(questions, dict) else enumerate(questions)
        return {name: self.prime_question(question) for name, question in items}


def create_stub_app(settings: Settings | None = None):
    settings = settings or Settings(model="stub", model_name="jevad-stub", device="cpu")
    return create_app(settings=settings, model=StubJeva())


app = create_stub_app()
