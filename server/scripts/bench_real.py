from __future__ import annotations

import json
import statistics
import time

import torch
from jeva import Choice, Jeva, Noul, Score

QUESTIONS = {
    "outage": Noul(instructions="Does this report an ongoing service outage?"),
    "team": Choice(
        instructions="Which team should handle this support ticket?",
        criteria={
            "billing": "Payment disputes, invoices, and refunds.",
            "engineering": "Service outages, software errors, and broken features.",
            "general": "All other customer support requests.",
        },
    ),
    "impact": Score(
        instructions="Rate the reported customer impact.",
        criteria=[
            "No customers are affected.",
            "Some customers are affected, but the service remains usable.",
            "All customers are blocked from using a core service.",
        ],
        values=[0, 1, 2],
    ),
}

BASE_STATE = (
    "The production login page returns HTTP 503. "
    "All customers are unable to sign in or complete purchases."
)


def timed(fn):
    started = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - started) * 1000


def summarize(name, samples):
    print(
        f"{name}: median={statistics.median(samples):.1f}ms "
        f"min={min(samples):.1f}ms max={max(samples):.1f}ms n={len(samples)}"
    )


def main():
    load_started = time.perf_counter()
    model = Jeva.from_pretrained(
        "./models/Jeva-4B",
        device="cuda",
        quantize="4bit",
        option_chunk_size=4,
        option_batch_tokens=4096,
    )
    print(
        f"load {time.perf_counter() - load_started:.1f}s "
        f"vram {torch.cuda.memory_allocated() / 2**30:.2f}GiB"
    )

    _, warmup = timed(lambda: model.system_one(BASE_STATE, QUESTIONS))
    print(f"warmup {warmup:.1f}ms")

    model.clear_cache()
    sequential = []
    for index in range(6):
        state = f"{BASE_STATE} Ticket {index}."
        _, elapsed = timed(lambda: model.system_one(state, QUESTIONS))
        sequential.append(elapsed)
    summarize("sequential 3-question request", sequential)

    model.clear_cache()
    cold_repeat = []
    for _ in range(3):
        response, elapsed = timed(lambda: model.system_one(BASE_STATE, QUESTIONS))
        cold_repeat.append(elapsed)
    summarize("repeated request (state+option cache)", cold_repeat)
    info = model.cache_info()
    print(
        "cache hits",
        json.dumps(
            {
                "states": info["states"]["hits"],
                "options": info["options"]["hits"],
                "sets": info["sets"]["hits"],
            }
        ),
    )

    batch_requests = [(f"{BASE_STATE} Batch {index}.", QUESTIONS) for index in range(4)]
    sequential_for_batch = []
    for state, questions in batch_requests:
        _, elapsed = timed(lambda: model.system_one(state, questions))
        sequential_for_batch.append(elapsed)
    batch_started = time.perf_counter()
    responses = model.system_one_many(batch_requests)
    batch_elapsed = (time.perf_counter() - batch_started) * 1000
    sequential_total = sum(sequential_for_batch)
    print(
        f"4 requests sequential total {sequential_total:.1f}ms | "
        f"batched total {batch_elapsed:.1f}ms | "
        f"speedup {sequential_total / batch_elapsed:.2f}x"
    )
    print(
        "batch encoded option tokens",
        [response.usage.encoded_option_tokens for response in responses],
    )

    try:
        model._compiled_decide = torch.compile(model.model.decide_from_option_cache, dynamic=False)
        _, compile_warmup = timed(lambda: model.system_one(BASE_STATE, QUESTIONS))
        compiled_samples = []
        for index in range(3):
            state = f"{BASE_STATE} Compiled {index}."
            _, elapsed = timed(lambda: model.system_one(state, QUESTIONS))
            compiled_samples.append(elapsed)
        summarize("compiled decision stack", compiled_samples)
        print(f"compile warmup {compile_warmup:.1f}ms")
    except Exception as exc:
        print("compile unavailable:", type(exc).__name__, exc)


if __name__ == "__main__":
    main()
