from __future__ import annotations

import time

import torch
from jeva import Choice, Jeva, Noul, Score

started = time.perf_counter()
model = Jeva.from_pretrained(
    "./models/Jeva-4B",
    device="cuda",
    quantize="4bit",
    option_chunk_size=4,
    option_batch_tokens=4096,
)
print(f"load {time.perf_counter() - started:.1f}s")
print(f"vram allocated {torch.cuda.memory_allocated() / 2**30:.2f} GiB")
print(f"vram reserved  {torch.cuda.memory_reserved() / 2**30:.2f} GiB")
print(f"lora_merged {model.lora_merged} quantize {model.quantize}")

started = time.perf_counter()
result = model.system_one(
    state=(
        "The production login page returns HTTP 503. "
        "All customers are unable to sign in or complete purchases."
    ),
    questions={
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
    },
)
print(f"first request {time.perf_counter() - started:.2f}s")
print("outage", result.answers["outage"].noul)
print("team", result.answers["team"].choice, result.answers["team"].probabilities)
print("impact", result.answers["impact"].score)
print(f"peak vram {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB")

started = time.perf_counter()
model.system_one(
    state="I was charged twice for the same order.",
    questions={"q": Noul(instructions="Is this about billing?")},
)
print(f"second request {time.perf_counter() - started:.2f}s")
print("cache", model.cache_info())
