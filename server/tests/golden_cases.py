from __future__ import annotations

import json

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

RANK_SPEC = {
    "state": "Customer was charged twice.",
    "candidates": {"refund": "Duplicate charge.", "cancel": "Cancel order."},
    "top_k": 1,
}

CHAT_SPEC = {
    "model": "jevad-stub",
    "messages": [{"role": "user", "content": json.dumps(DECISION_SPEC)}],
}

STREAM_SPEC = {
    "model": "jevad-stub",
    "stream": True,
    "stream_options": {"include_usage": True},
    "messages": [{"role": "user", "content": json.dumps(DECISION_SPEC)}],
}


def _stream_events(client, path, payload) -> list[dict]:
    events: list[dict] = []
    with client.stream("POST", path, json=payload) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data == "[DONE]":
                continue
            events.append(json.loads(data))
    return events


def collect(client) -> dict:
    return {
        "healthz": client.get("/healthz").json(),
        "models": client.get("/v1/models").json(),
        "model": client.get("/v1/models/jevad-stub").json(),
        "decide": client.post("/v1/decide", json=DECISION_SPEC).json(),
        "rank": client.post("/v1/rank", json=RANK_SPEC).json(),
        "cache": client.get("/v1/cache").json(),
        "prime": client.post(
            "/v1/cache/prime",
            json={
                "candidates": {"refund": "Duplicate charge.", "cancel": "Cancel order."},
            },
        ).json(),
        "chat": client.post("/v1/chat/completions", json=CHAT_SPEC).json(),
        "completions": client.post(
            "/v1/completions", json={"prompt": json.dumps(DECISION_SPEC)}
        ).json(),
        "chat_stream": _stream_events(client, "/v1/chat/completions", STREAM_SPEC),
        "error_validation": client.post("/v1/decide", json={"state": "x", "questions": {}}).json(),
        "error_not_found": client.get("/v1/models/unknown").json(),
    }
