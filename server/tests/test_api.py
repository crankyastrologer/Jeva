from __future__ import annotations

import asyncio
import json

import pytest
from fastapi.testclient import TestClient

from jeva import Choice, Noul, Score
from jeva.types import Answer, DecisionResponse, RankedItem, RankResult, Usage
from jevad.app import create_app
from jevad.config import Settings

DECISION_SPEC = {
    "state": "The login page returns HTTP 503 for all customers.",
    "questions": {
        "outage": {"type": "noul", "instructions": "Is this an outage?"},
        "team": {
            "type": "choice",
            "instructions": "Which team handles this?",
            "criteria": {"billing": "Payments.", "engineering": "Outages."},
        },
        "impact": {
            "type": "score",
            "instructions": "Rate the impact.",
            "criteria": ["None", "Some", "All"],
            "values": [0, 1, 2],
        },
    },
}


class FakeJeva:
    def __init__(self) -> None:
        self.calls: list[tuple] = []

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
        if isinstance(question, Choice):
            probabilities = {label: 1.0 / len(labels) for label in labels}
            return Answer(
                kind="choice",
                probabilities=probabilities,
                conditional_probabilities=probabilities,
                null_probability=0.0,
                choice=labels[0],
                confidence=1.0 / len(labels),
            )
        assert isinstance(question, Score)
        probabilities = {label: 1.0 / len(labels) for label in labels}
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
        labels = list(candidates) if isinstance(candidates, dict) else list(candidates)
        ranking = [
            RankedItem(label=label, probability=1.0 / len(labels), index=index)
            for index, label in enumerate(labels)
        ]
        if top_k is not None:
            ranking = ranking[:top_k]
        return RankResult(ranking=ranking, null_probability=0.0, usage=self._usage(), latency_ms=3.0)

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

    def prime(self, questions):
        items = questions.items() if isinstance(questions, dict) else enumerate(questions)
        return {name: self.prime_options([], kind="choice") for name, _ in items}

    def cache_info(self):
        return {
            "mode": "hybrid",
            "options": {"entries": 2, "bytes": 128},
            "sets": {"entries": 1, "bytes": 64},
        }

    def clear_cache(self):
        self.calls.append(("clear_cache",))


@pytest.fixture
def settings():
    return Settings(
        model="stub",
        model_name="jevad-stub",
        device="cpu",
        cache="hybrid",
        queue_size=4,
        request_timeout=5.0,
    )


@pytest.fixture
def client(settings):
    fake = FakeJeva()
    app = create_app(settings=settings, model=fake)
    with TestClient(app) as test_client:
        test_client.fake = fake
        yield test_client


def test_healthz(client):
    response = client.get("/healthz")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model"] == "jevad-stub"
    assert body["queue"] == {"depth": 0, "capacity": 4}


def test_models(client):
    response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json()["data"][0]["id"] == "jevad-stub"
    assert client.get("/v1/models/jevad-stub").status_code == 200
    assert client.get("/v1/models/unknown").status_code == 404


def test_decide_named(client):
    response = client.post("/v1/decide", json=DECISION_SPEC)
    assert response.status_code == 200
    body = response.json()
    assert body["answers"]["outage"]["noul"] == 0.8
    assert body["answers"]["team"]["choice"] == "billing"
    assert body["answers"]["impact"]["score"] == pytest.approx(1.0)
    assert body["usage"]["input_tokens"] == 12
    state, questions = client.fake.calls[0][1], client.fake.calls[0][2]
    assert state.startswith("The login page")
    assert isinstance(questions["team"], Choice)


def test_decide_list(client):
    spec = {
        "state": {"ticket": 42, "text": "Payment failed twice."},
        "questions": [
            {"type": "noul", "instructions": "Is this about billing?"},
        ],
    }
    response = client.post("/v1/decide", json=spec)
    assert response.status_code == 200
    assert isinstance(response.json()["answers"], list)
    assert client.fake.calls[0][1] == {"ticket": 42, "text": "Payment failed twice."}


def test_decide_rejects_bad_json(client):
    response = client.post("/v1/decide", json={"state": "x", "questions": {}})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_decide_rejects_unknown_type(client):
    spec = {
        "state": "x",
        "questions": {"q": {"type": "ranking", "instructions": "x"}},
    }
    response = client.post("/v1/decide", json=spec)
    assert response.status_code == 400


def test_chat_completion(client):
    payload = {
        "model": "jevad-stub",
        "messages": [{"role": "user", "content": json.dumps(DECISION_SPEC)}],
    }
    response = client.post("/v1/chat/completions", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["finish_reason"] == "stop"
    content = json.loads(body["choices"][0]["message"]["content"])
    assert content["answers"]["team"]["choice"] == "billing"
    assert body["usage"] == {"prompt_tokens": 12, "completion_tokens": 0, "total_tokens": 12}


def test_chat_completion_requires_user_message(client):
    response = client.post(
        "/v1/chat/completions",
        json={"messages": [{"role": "system", "content": "hi"}]},
    )
    assert response.status_code == 400


def test_chat_completion_stream(client):
    payload = {
        "model": "jevad-stub",
        "stream": True,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": json.dumps(DECISION_SPEC)}],
    }
    pieces: list[str] = []
    saw_done = False
    saw_usage = False
    with client.stream("POST", "/v1/chat/completions", json=payload) as response:
        assert response.status_code == 200
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line.removeprefix("data: ")
            if data == "[DONE]":
                saw_done = True
                continue
            event = json.loads(data)
            assert event["object"] == "chat.completion.chunk"
            if event["usage"]:
                saw_usage = True
            for choice in event["choices"]:
                pieces.append(choice["delta"].get("content") or "")
    assert saw_done
    assert saw_usage
    assert json.loads("".join(pieces))["answers"]["impact"]["score"] == pytest.approx(1.0)


def test_completions(client):
    response = client.post("/v1/completions", json={"prompt": json.dumps(DECISION_SPEC)})
    assert response.status_code == 200
    body = response.json()
    assert body["object"] == "text_completion"
    assert json.loads(body["choices"][0]["text"])["answers"]["outage"]["noul"] == 0.8


def test_completions_rejects_plain_text(client):
    response = client.post("/v1/completions", json={"prompt": "hello"})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"


def test_rank(client):
    payload = {
        "state": "Customer was charged twice.",
        "candidates": {"refund": "Duplicate charge.", "cancel": "Cancel order."},
        "top_k": 1,
    }
    response = client.post("/v1/rank", json=payload)
    assert response.status_code == 200
    body = response.json()
    assert len(body["ranking"]) == 1
    assert body["ranking"][0]["index"] == 0
    assert body["null_probability"] == 0.0


def test_cache_endpoints(client):
    info = client.get("/v1/cache")
    assert info.status_code == 200
    assert info.json()["options"]["entries"] == 2
    assert "states" in info.json()
    cleared = client.post("/v1/cache/clear")
    assert cleared.status_code == 200
    assert ("clear_cache",) in client.fake.calls


def test_engine_batches_jobs():
    from jevad.runtime import InferenceEngine

    batches: list[int] = []

    def executor(jobs):
        batches.append(len(jobs))
        return list(jobs)

    async def scenario():
        engine = InferenceEngine(executor, queue_size=8, batch_window=0.05, max_batch=4)
        await engine.start()
        try:
            return await asyncio.gather(*[engine.submit(index) for index in range(4)])
        finally:
            await engine.stop()

    assert asyncio.run(scenario()) == [0, 1, 2, 3]
    assert max(batches) > 1


def test_prime_endpoint(client):
    response = client.post(
        "/v1/cache/prime",
        json={"candidates": {"refund": "Duplicate charge.", "cancel": "Unshipped order."}},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["primed"]["options"] == 2
    assert "cache" in body
    assert any(call[0] == "prime_options" for call in client.fake.calls)


def test_prime_endpoint_requires_one_source(client):
    assert client.post("/v1/cache/prime", json={}).status_code == 400
    both = {"candidates": ["a"], "questions": {"q": {"type": "noul", "instructions": "x"}}}
    assert client.post("/v1/cache/prime", json=both).status_code == 400


def test_metrics(client):
    client.post("/v1/decide", json=DECISION_SPEC)
    response = client.get("/metrics")
    assert response.status_code == 200
    text = response.text
    assert 'jeva_requests_total{endpoint="/v1/decide",status="200"} 1' in text
    assert "jeva_queue_depth 0" in text


def test_auth(settings):
    protected = Settings(**{**settings.__dict__, "api_key": "secret"})
    app = create_app(settings=protected, model=FakeJeva())
    with TestClient(app) as test_client:
        assert test_client.get("/healthz").status_code == 200
        assert test_client.get("/metrics").status_code == 200
        denied = test_client.get("/v1/models")
        assert denied.status_code == 401
        assert denied.json()["error"]["type"] == "authentication_error"
        allowed = test_client.get("/v1/models", headers={"Authorization": "Bearer secret"})
        assert allowed.status_code == 200


def test_queue_full(client, settings):
    response = client.post("/v1/decide", json=DECISION_SPEC)
    assert response.status_code == 200


def test_openapi(client):
    assert client.get("/openapi.json").status_code == 200
