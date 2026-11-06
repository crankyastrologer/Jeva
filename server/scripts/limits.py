from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
import uuid
from pathlib import Path

import httpx


class VramSampler(threading.Thread):
    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.samples: list[int] = []
        self.stop = threading.Event()

    def run(self) -> None:
        while not self.stop.is_set():
            try:
                out = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                self.samples.append(int(out.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(self.interval)

    def peak(self) -> int:
        return max(self.samples) if self.samples else 0


def candidate_bank(count: int, salt: str) -> dict[str, str]:
    return {
        f"option_{index}": (
            f"Policy {salt} entry {index}: handles category {index} requests "
            "according to the support runbook and escalation matrix."
        )
        for index in range(count)
    }


def long_state(repeats: int, salt: str) -> str:
    sentence = "Customer reports an issue with the checkout flow after a recent update. "
    return f"{sentence * repeats} Reference {salt}."


def cold_samples(client: httpx.Client, path: str, make_payload, repeats: int = 3) -> dict:
    salt = uuid.uuid4().hex[:6]
    samples = []
    first_usage = {}
    for index in range(repeats):
        payload = make_payload(index, salt)
        started = time.perf_counter()
        response = client.post(path, json=payload)
        elapsed = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            return {"status": response.status_code, "body": response.text[:300], "salt": salt}
        body = response.json()
        samples.append(round(elapsed, 1))
        if index == 0:
            first_usage = body.get("usage", {})
    return {
        "status": 200,
        "salt": salt,
        "samples_ms": samples,
        "median_ms": round(statistics.median(samples), 1),
        "first_usage": first_usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8079")
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    results = {}
    with httpx.Client(base_url=args.url, timeout=1800.0) as client:
        print("healthz", client.get("/healthz").status_code)

        results["option_scaling"] = []
        for count in (4, 16, 64, 256, 1024, 2048):
            sampler = VramSampler()
            sampler.start()
            result = cold_samples(
                client,
                "/v1/decide",
                lambda index, salt, count=count: {
                    "state": f"The login page is down. Case {salt}-{index}.",
                    "questions": {
                        "team": {
                            "type": "choice",
                            "instructions": "Which runbook entry applies?",
                            "criteria": candidate_bank(count, salt),
                        }
                    },
                },
            )
            sampler.stop.set()
            sampler.join()
            result.update({"options": count, "peak_vram_mib": sampler.peak()})
            results["option_scaling"].append(result)
            print(
                f"options={count:<5} status={result['status']} "
                f"samples={result.get('samples_ms')} "
                f"tokens={result.get('first_usage', {}).get('encoded_option_tokens')} "
                f"vram={result['peak_vram_mib']}MiB"
            )

        results["state_scaling"] = []
        for repeats in (5, 50, 200, 600, 700):
            sampler = VramSampler()
            sampler.start()
            result = cold_samples(
                client,
                "/v1/decide",
                lambda index, salt, repeats=repeats: {
                    "state": long_state(repeats, f"{salt}-{index}"),
                    "questions": {"q": {"type": "noul", "instructions": "Is this an outage?"}},
                },
            )
            sampler.stop.set()
            sampler.join()
            result.update({"repeats": repeats, "peak_vram_mib": sampler.peak()})
            results["state_scaling"].append(result)
            print(
                f"repeats={repeats:<4} status={result['status']} "
                f"samples={result.get('samples_ms')} "
                f"tokens={result.get('first_usage', {}).get('encoded_state_tokens')} "
                f"vram={result['peak_vram_mib']}MiB"
            )

        results["cache"] = client.get("/v1/cache").json()

    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results, indent=2) + "\n")
        print("wrote", path)


if __name__ == "__main__":
    main()
