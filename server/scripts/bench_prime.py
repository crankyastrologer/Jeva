from __future__ import annotations

import statistics
import time

import torch
from jeva import Choice, Jeva

STATE = "The production login page returns HTTP 503 for all customers."
BANK = {
    "refund": "Duplicate payments and refunds. Verify both transactions settled for the same order, then refund the duplicate.",
    "cancellation": "Cancel an order before dispatch. Stop fulfillment if the package was not handed to the carrier.",
    "delivery": "Shipment tracking and delays. Retrieve carrier, tracking number, latest scan, and estimated delivery.",
    "account": "Customer contact information updates. Verify identity, then change email or phone and notify the customer.",
    "billing": "Invoices and payment method questions. Do not request full card numbers in support conversations.",
    "returns": "Return procedure after delivery. Explain the return window and generate a return label.",
    "warranty": "Warranty claims for defective products. Collect serial number and proof of purchase.",
    "technical": "Product defects and integration errors. Reproduce the issue and escalate to engineering when needed.",
}


def run(model, tag, instructions, count=5):
    samples = []
    encoded = []
    for index in range(count):
        started = time.perf_counter()
        response = model.system_one(
            f"{STATE} {tag} request {index}.",
            {"q": Choice(instructions, BANK)},
        )
        samples.append((time.perf_counter() - started) * 1000)
        encoded.append(response.answers["q"].usage.encoded_option_tokens)
    print(
        f"{tag}: median={statistics.median(samples):.0f}ms "
        f"samples={[round(value) for value in samples]} "
        f"encoded_option_tokens={encoded}"
    )
    return samples


def main():
    model = Jeva.from_pretrained(
        "./models/Jeva-4B",
        device="cuda",
        quantize="4bit",
        option_chunk_size=4,
        option_batch_tokens=4096,
    )
    model.system_one(f"{STATE} warmup.", {"q": Choice("Warm up.", {"a": "A.", "b": "B."})})
    print(f"vram {torch.cuda.memory_allocated() / 2**30:.2f}GiB")
    instructions = "Which team should handle this support ticket?"

    model.clear_cache()
    run(model, "cold (no priming)", instructions)

    model.clear_cache()
    primed = model.prime_options([f"{key}: {text}" for key, text in BANK.items()], kind="choice")
    print(
        f"prime: {primed['encoded_tokens']} tokens in {primed['latency_ms']:.0f}ms "
        f"set_cached={primed['set_cached']}"
    )
    run(model, "primed, same question", instructions)

    model.cache = "none"
    run(model, "cache none", instructions)

    model.cache = "hybrid"
    model.clear_cache()
    model.prime_options([f"{key}: {text}" for key, text in BANK.items()], kind="choice")
    for variant in (
        "Triage the ticket.",
        "Route this customer request.",
        "Select the owning team.",
    ):
        run(model, f"primed, new instructions ({variant[:18]}...)", variant, count=2)


if __name__ == "__main__":
    main()
