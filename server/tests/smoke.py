from __future__ import annotations

import json
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8077"
SPEC = json.loads(Path(__file__).with_name("spec.json").read_text())

with httpx.Client(base_url=BASE, timeout=30) as client:
    health = client.get("/healthz")
    print("healthz", health.status_code, health.json()["status"])

    models = client.get("/v1/models").json()
    print("models", [card["id"] for card in models["data"]])

    decide = client.post("/v1/decide", json=SPEC)
    print("decide", decide.status_code)
    answers = decide.json()["answers"]
    print("  noul", answers["outage"]["noul"], "choice", answers["team"]["choice"], "score", answers["impact"]["score"])

    rank = client.post("/v1/rank", json={
        "state": "charged twice",
        "candidates": {"refund": "duplicate charge", "cancel": "unshipped order"},
        "top_k": 1,
    })
    print("rank", rank.status_code, rank.json()["ranking"][0]["label"])

    pieces: list[str] = []
    events = 0
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "jevad-stub",
        "stream": True,
        "messages": [{"role": "user", "content": json.dumps(SPEC)}],
    }) as stream:
        for line in stream.iter_lines():
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                continue
            events += 1
            for choice in json.loads(data)["choices"]:
                pieces.append(choice["delta"].get("content") or "")
    payload = json.loads("".join(pieces))
    print("stream", stream.status_code, "events", events, "choice", payload["answers"]["team"]["choice"])

    metrics = client.get("/metrics")
    print("metrics", metrics.status_code, "decide counter", 'endpoint="/v1/decide"' in metrics.text)
