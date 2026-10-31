from __future__ import annotations

import json
import time

import httpx

BASE = "http://127.0.0.1:8079"
BANK = {
    "refund": "Duplicate payments and refunds.",
    "cancellation": "Cancel an order before dispatch.",
    "delivery": "Shipment tracking and delays.",
}
STATE = "The production login page returns HTTP 503 for all customers."


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

        primed = client.post("/v1/cache/prime", json={"candidates": BANK})
        primed.raise_for_status()
        print("prime", primed.status_code, primed.json()["primed"]["encoded_tokens"], "tokens")

        for instructions in ("Route this ticket.", "Which team owns it?", "Pick the owning team."):
            response = client.post(
                "/v1/decide",
                json={
                    "state": STATE,
                    "questions": {
                        "team": {
                            "type": "choice",
                            "instructions": instructions,
                            "criteria": BANK,
                        }
                    },
                },
            )
            response.raise_for_status()
            answer = response.json()["answers"]["team"]
            print(
                f"{instructions!r}: {answer['choice']} "
                f"encoded={answer['usage']['encoded_option_tokens']} "
                f"cached={answer['usage']['cached_option_tokens']}"
            )

        print("cache", json.dumps(client.get("/v1/cache").json()["options"]))


if __name__ == "__main__":
    main()
