from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
import uuid
from collections import Counter
from pathlib import Path

import httpx
import numpy as np

BASE_STATE = (
    "The production login page returns HTTP 503. "
    "All customers are unable to sign in or complete purchases."
)

CHOICE = {
    "type": "choice",
    "instructions": "Which team should handle this support ticket?",
    "criteria": {
        "billing": "Payment disputes, invoices, and refunds.",
        "engineering": "Service outages, software errors, and broken features.",
        "general": "All other customer support requests.",
    },
}

THREE_QUESTIONS = {
    "outage": {"type": "noul", "instructions": "Does this report an ongoing service outage?"},
    "team": CHOICE,
    "impact": {
        "type": "score",
        "instructions": "Rate the reported customer impact.",
        "criteria": ["No customers affected.", "Some affected.", "All blocked."],
        "values": [0, 1, 2],
    },
}

CANDIDATES_8 = {
    f"policy_{index}": f"Policy {index}: handles category {index} requests per the support runbook."
    for index in range(8)
}
CANDIDATES_32 = {
    f"policy_{index}": f"Policy {index}: handles category {index} requests per the support runbook."
    for index in range(32)
}


def unique_state(index: int) -> str:
    return f"{BASE_STATE} Case {index} {uuid.uuid4().hex[:8]}."


def decide_payload(index: int) -> dict:
    return {"state": unique_state(index), "questions": THREE_QUESTIONS}


def noul_payload(index: int) -> dict:
    return {
        "state": unique_state(index),
        "questions": {"q": {"type": "noul", "instructions": "Is this a service outage?"}},
    }


def rank_payload(index: int, candidates: dict[str, str]) -> dict:
    return {"state": unique_state(index), "candidates": candidates}


async def run_phase(url: str, mode: str, concurrency: int, duration: float) -> dict:
    limits = httpx.Limits(
        max_connections=concurrency + 16,
        max_keepalive_connections=concurrency + 16,
    )
    latencies: list[float] = []
    server_latencies: list[float] = []
    statuses: Counter = Counter()
    encoded_state = 0
    encoded_options = 0
    cached_options = 0
    counter = 0

    async with httpx.AsyncClient(base_url=url, timeout=600.0, limits=limits) as client:
        stop_at = time.perf_counter() + duration

        async def worker() -> None:
            nonlocal counter, encoded_state, encoded_options, cached_options
            while time.perf_counter() < stop_at:
                counter += 1
                index = counter
                if mode == "noul":
                    path, payload = "/v1/decide", noul_payload(index)
                elif mode == "rank8":
                    path, payload = "/v1/rank", rank_payload(index, CANDIDATES_8)
                elif mode == "rank32":
                    path, payload = "/v1/rank", rank_payload(index, CANDIDATES_32)
                else:
                    path, payload = "/v1/decide", decide_payload(index)
                started = time.perf_counter()
                try:
                    response = await client.post(path, json=payload)
                except Exception as exc:
                    statuses[type(exc).__name__] += 1
                    continue
                latencies.append((time.perf_counter() - started) * 1000)
                statuses[str(response.status_code)] += 1
                if response.status_code != 200:
                    continue
                body = response.json()
                server_latencies.append(body.get("latency_ms", 0.0))
                usage = body.get("usage", {})
                encoded_state += usage.get("encoded_state_tokens", 0)
                encoded_options += usage.get("encoded_option_tokens", 0)
                cached_options += usage.get("cached_option_tokens", 0)

        started = time.perf_counter()
        await asyncio.gather(*[worker() for _ in range(concurrency)])
        elapsed = time.perf_counter() - started

    successes = statuses.get("200", 0)
    return {
        "mode": mode,
        "concurrency": concurrency,
        "duration_s": round(elapsed, 2),
        "requests": sum(statuses.values()),
        "success": successes,
        "statuses": dict(statuses),
        "throughput_rps": round(successes / elapsed, 2),
        "latency_ms": {
            "mean": round(statistics.fmean(latencies), 1) if latencies else None,
            "p50": round(float(np.percentile(latencies, 50)), 1) if latencies else None,
            "p90": round(float(np.percentile(latencies, 90)), 1) if latencies else None,
            "p95": round(float(np.percentile(latencies, 95)), 1) if latencies else None,
            "p99": round(float(np.percentile(latencies, 99)), 1) if latencies else None,
            "max": round(max(latencies), 1) if latencies else None,
        },
        "server_latency_ms": {
            "p50": round(float(np.percentile(server_latencies, 50)), 1)
            if server_latencies
            else None,
            "p95": round(float(np.percentile(server_latencies, 95)), 1)
            if server_latencies
            else None,
        },
        "tokens": {"state": encoded_state, "options": encoded_options, "cached": cached_options},
    }


def print_phase(result: dict) -> None:
    lat = result["latency_ms"]
    print(
        f"{result['mode']:<8} c={result['concurrency']:<3} "
        f"rps={result['throughput_rps']:<7} "
        f"p50={lat['p50']:<8} p95={lat['p95']:<8} p99={lat['p99']:<8} "
        f"max={lat['max']:<8} ok={result['success']:<5} statuses={result['statuses']}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8079")
    parser.add_argument("--modes", default="noul,decide3,rank8")
    parser.add_argument("--concurrency", default="1,2,4,8,16")
    parser.add_argument("--duration", type=float, default=15.0)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    modes = args.modes.split(",")
    levels = [int(value) for value in args.concurrency.split(",")]

    async with httpx.AsyncClient(base_url=args.url, timeout=600.0) as client:
        health = await client.get("/healthz")
        print("healthz", health.status_code, health.json()["status"])
        for index in range(args.warmup):
            response = await client.post("/v1/decide", json=decide_payload(index))
            response.raise_for_status()
        print(f"warmup done ({args.warmup} requests)")

    results = []
    for mode in modes:
        for level in levels:
            result = await run_phase(args.url, mode, level, args.duration)
            print_phase(result)
            results.append(result)

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n")
        print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
