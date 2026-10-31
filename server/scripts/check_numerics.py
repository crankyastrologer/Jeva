from __future__ import annotations

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
STATE = (
    "The production login page returns HTTP 503. "
    "All customers are unable to sign in or complete purchases."
)


def show(tag, result):
    print(
        tag,
        "noul",
        round(result.answers["outage"].noul, 6),
        "team",
        result.answers["team"].choice,
        round(result.answers["team"].probabilities["engineering"], 6),
        "score",
        round(result.answers["impact"].score, 6),
    )


def main():
    model = Jeva.from_pretrained(
        "./models/Jeva-4B",
        device="cuda",
        quantize="4bit",
        option_chunk_size=4,
        option_batch_tokens=4096,
    )
    model.system_one(f"{STATE} warm", QUESTIONS)

    model.clear_cache()
    model.state_chunk_size = 1
    show("chunk1 exact", model.system_one(STATE, QUESTIONS))

    model.clear_cache()
    model.state_chunk_size = 8
    show("chunk8 exact", model.system_one(STATE, QUESTIONS))

    model.clear_cache()
    show("chunk8 warmup-suffix", model.system_one(f"{STATE} Warmup.", QUESTIONS))

    responses = model.system_one_many(
        [
            (STATE, QUESTIONS),
            (f"{STATE} Other.", QUESTIONS),
        ]
    )
    show("batched exact", responses[0])
    show("batched other", responses[1])


if __name__ == "__main__":
    main()
