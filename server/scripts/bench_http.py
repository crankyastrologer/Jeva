from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

BASE = "http://127.0.0.1:8079"
SPEC = json.loads(Path(__file__).with_name("spec.json").read_text())
STATE = SPEC["state"]


def post(client, state):
    started = time.perf_counter()
    response = client.post("/v1/decide", json={**SPEC, "state": state})
    elapsed = (time.perf_counter() - started) * 1000
    response.raise_for_status()
    return response.json(), elapsed


def sequential(client, tag, count=4):
    samples = []
    for index in range(count):
        _, elapsed = post(client, f"{STATE} {tag} sequential {index}.")
        samples.append(elapsed)
    return samples


def concurrent(client, tag, count=4):
    states = [f"{STATE} {tag} concurrent {index}." for index in range(count)]
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=count) as pool:
        results = list(pool.map(lambda state: post(client, state), states))
    total = (time.perf_counter() - started) * 1000
    return total, [elapsed for _, elapsed in results]


def main():
    with httpx.Client(base_url=BASE, timeout=600) as client:
        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if client.get("/healthz").json()["status"] == "ok":
                    break
            except Exception:
                pass
            time.sleep(2)

        body, warmup = post(client, f"{STATE} Warmup.")
        print(
            f"warmup {warmup:.0f}ms outage={body['answers']['outage']['noul']:.4f} "
            f"team={body['answers']['team']['choice']} score={body['answers']['impact']['score']:.2f}"
        )
        sequential(client, "prime", count=8)

        for round_index in range(2):
            sequential_samples = sequential(client, f"r{round_index}")
            concurrent_total, concurrent_samples = concurrent(client, f"r{round_index}")
            print(
                f"round {round_index}: sequential median={statistics.median(sequential_samples):.0f}ms "
                f"total={sum(sequential_samples):.0f}ms | concurrent total={concurrent_total:.0f}ms "
                f"median={statistics.median(concurrent_samples):.0f}ms | "
                f"gain={sum(sequential_samples) / concurrent_total:.2f}x"
            )

        metrics = client.get("/metrics").text
        for line in metrics.splitlines():
            if line.startswith(("jeva_batches_total", "jeva_batch_jobs_total")):
                print(line)


if __name__ == "__main__":
    main()
