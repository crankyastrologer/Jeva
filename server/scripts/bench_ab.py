from __future__ import annotations

import statistics
import time

from jeva import Choice, Jeva, Noul, Score

QUESTIONS = {
    "outage": Noul(instructions="Does this report an ongoing service outage?"),
    "team": Choice(
        instructions="Which team should handle this support ticket?",
        criteria={"billing": "Payments.", "engineering": "Outages.", "general": "Other."},
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
    for chunk_size in (1, 8, 1, 8):
        model.state_chunk_size = chunk_size
        model.clear_cache()
        samples = []
        for index in range(5):
            started = time.perf_counter()
            model.system_one(f"{STATE} ab {chunk_size} {index}.", QUESTIONS)
            samples.append((time.perf_counter() - started) * 1000)
        print(
            f"state_chunk_size={chunk_size} median={statistics.median(samples):.1f}ms "
            f"samples={[round(value) for value in samples]}"
        )


if __name__ == "__main__":
    main()
