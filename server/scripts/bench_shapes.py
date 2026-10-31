from __future__ import annotations

import time

import torch
from jeva import Choice, Jeva, Noul, Score

QUESTIONS = {
    "outage": Noul(instructions="Does this report an ongoing service outage?"),
    "team": Choice(
        instructions="Which team should handle this support ticket?",
        criteria={"billing": "Payment disputes.", "engineering": "Outages.", "general": "Other."},
    ),
    "impact": Score(
        instructions="Rate the customer impact.",
        criteria=["None", "Some", "All"],
        values=[0, 1, 2],
    ),
}
STATE = "The production login page returns HTTP 503 for all customers."


def main():
    model = Jeva.from_pretrained(
        "./models/Jeva-4B",
        device="cuda",
        quantize="4bit",
        option_chunk_size=4,
        option_batch_tokens=4096,
    )
    model.system_one(f"{STATE} warmup", QUESTIONS)
    print(f"vram {torch.cuda.memory_allocated() / 2**30:.2f}GiB")

    for size in (1, 2, 3, 4, 6, 8):
        requests = [(f"{STATE} shape {size} request {index}.", QUESTIONS) for index in range(size)]
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            model.system_one_many(requests)
            samples.append((time.perf_counter() - started) * 1000)
        print(
            f"batch={size} requests ({size * len(QUESTIONS)} state rows): "
            f"samples={[round(value) for value in samples]} "
            f"per_request={round(min(samples) / size)}ms"
        )

    print("cache", model.cache_info()["states"])
    print(f"peak vram {torch.cuda.max_memory_allocated() / 2**30:.2f}GiB")


if __name__ == "__main__":
    main()
