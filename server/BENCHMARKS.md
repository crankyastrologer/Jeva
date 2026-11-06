# Server benchmarks

Hard numbers for `jevad` on the machine this project was developed on. Raw
results are in `server/scripts/results/`; the generators are
`server/scripts/loadtest.py` and `server/scripts/limits.py`.

## Hardware and configuration

| Item | Value |
| :-- | :-- |
| GPU | NVIDIA RTX 4060 Laptop, 8,188 MiB, 48 W cap |
| Host | WSL2 Ubuntu 24.04, kernel 6.6.87.2, 7 GB RAM |
| Driver | 617.42 (KMD), CUDA 13.4 UMD |
| Torch | 2.9.1+cu128 |
| Model | Jeva-4B (`qwen3_5` backbone, 8.8 GB bf16) |
| Quantization | bitsandbytes NF4, 4-bit, head in fp32 |
| Server | `JEVA_OPTION_CHUNK_SIZE=4`, `JEVA_OPTION_BATCH_TOKENS=4096`, `JEVA_MAX_BATCH=8`, `JEVA_BATCH_WINDOW_MS=5` unless stated |
| Transport | HTTP over localhost, JSON, no TLS |

Idle after load: **4.3 GiB** VRAM. Peak under load: **5.9 GiB**.

## Proof the GPU is actually used

Some readers looked at `nvidia-smi` between requests and saw 0%. That is
expected: the model is 4-bit and requests are short prefill bursts. Sampling
every 500 ms during a continuous load run:

| Metric | Value |
| :-- | :-- |
| Average GPU utilization | **82.1%** |
| Maximum utilization | **100%** |
| Maximum board power | **48 W** (at cap) |
| Samples | 58 (29 s of load) |

The server process also holds 5.9 GiB and appears in
`nvidia-smi --query-compute-apps`. A cold 7,828-token state takes 4.5 s, and
2,048 options take 83 s to encode; this is not CPU work.

## Cold start

| Phase | Time |
| :-- | :-- |
| First ever process start (cold filesystem, kernel cache absent) | **57 s** |
| Subsequent process starts (OS page cache warm) | **28–30 s** |
| First request, kernel cache absent | **39 s** (measured earlier) |
| First request, kernel cache warm | **2.1 s** |

The 57 s includes loading and 4-bit quantizing an 8.8 GB checkpoint. Triton/FLA
kernel caches are written on the first request and reused by later processes,
which is why the first request drops from tens of seconds to ~2 s.

## Latency and throughput (closed loop)

Method: `--concurrency N` workers each send requests back to back for 12 s.
Every request has a unique state, so state encodings are always cold. Options
are warm after warmup because almost every real workload reuses a candidate
bank. Latency is client-side wall time including HTTP and JSON.

### `noul` (1 question)

| Concurrency | req/s | p50 ms | p95 ms | p99 ms |
| :-- | --: | --: | --: | --: |
| 1 | 7.75 | 121.7 | 165.7 | 201.0 |
| 2 | 15.94 | 121.6 | 144.3 | 157.2 |
| 4 | 21.10 | 189.8 | 197.2 | 199.0 |
| 8 | 20.17 | 324.7 | 1031.6 | 1226.3 |
| 16 | 24.29 | 634.9 | 904.5 | 993.9 |

### `decide3` (Noul + 4-way Choice + 3-level Score)

| Concurrency | req/s | p50 ms | p95 ms | p99 ms |
| :-- | --: | --: | --: | --: |
| 1 | 6.01 | 166.0 | 176.2 | 186.6 |
| 2 | 8.01 | 249.0 | 255.9 | 261.9 |
| 4 | 8.20 | 488.4 | 493.7 | 498.4 |
| 8 | 8.60 | 904.7 | 1148.8 | 1156.8 |
| 16 | 7.19 | 1839.6 | 4095.8 | 4096.6 |

### `rank` with 8 candidates

| Concurrency | req/s | p50 ms | p95 ms | p99 ms |
| :-- | --: | --: | --: | --: |
| 1 | 7.27 | 127.0 | 198.7 | 243.6 |
| 2 | 8.02 | 239.4 | 323.1 | 356.2 |
| 4 | 7.37 | 536.3 | 587.9 | 591.3 |
| 8 | 7.48 | 1064.6 | 1381.2 | 1499.8 |
| 16 | 7.11 | 2197.5 | 2440.6 | 2443.9 |

`/v1/rank` is deliberately not batched. Throughput stays flat around 7.3 req/s
and latency grows linearly with concurrency, because the single worker executes
one rank request at a time. Batched decide traffic is the path that scales.

## Batching A/B

Same workload, `JEVA_MAX_BATCH=1` (batching off) versus 8 (on):

| Workload | Concurrency | Batch 1 req/s | Batch 8 req/s | Gain | Batch 1 p50 | Batch 8 p50 |
| :-- | --: | --: | --: | --: | --: | --: |
| `noul` | 4 | 8.91 | 21.10 | **2.4×** | 443.2 ms | 189.8 ms |
| `noul` | 8 | 9.30 | 20.17 | **2.2×** | 856.2 ms | 324.7 ms |
| `decide3` | 4 | 6.31 | 8.20 | **1.3×** | 632.1 ms | 488.4 ms |
| `decide3` | 8 | 6.17 | 8.60 | **1.4×** | 1280.8 ms | 904.7 ms |

The gain is largest for single-question requests, where per-request overhead is
a bigger share of the work. For three-question requests the GPU is already busy,
so batching mostly removes launch overhead. This is not the 10–50× that
continuous batching gives generation workloads, and it should not be sold as
such.

## Option-count scaling

Cold numbers: a fresh option bank per sample (options always encoded). Warm
numbers reuse that bank. Each sample also has a fresh state.

| Options | Cold p50 | Encoded option tokens | Second call | Third call |
| --: | --: | --: | --: | --: |
| 4 | 536 ms | 132 | 121 ms | 114 ms |
| 16 | 758 ms | 578 | 124 ms | 124 ms |
| 64 | 2,687 ms | 2,338 | 138 ms | 124 ms |
| 256 | 9,971 ms | 9,910 | 126 ms | 148 ms |
| 1,024 | 40,266 ms | 40,702 | 140 ms | 164 ms |
| 2,048 | 82,782 ms | 86,782 | 164 ms | 150 ms |

With `option_chunk_size=4` (chosen for an 8 GB card), cold option encoding runs
at roughly 25 options/second. Once a bank is primed, adding candidates is
almost free at the decision head; the second and third calls are ~120–160 ms no
matter the bank size. **Priming is mandatory for large banks.** A larger GPU can
raise `option_chunk_size` and `option_batch_tokens` substantially, which is the
main knob for this curve.

## State-length scaling

Cold, one Noul question, state repeated to a target size:

| Repeats | Encoded state tokens | p50 latency |
| --: | --: | --: |
| 5 | 93 | 141 ms (118–116 ms after warmup) |
| 50 | 679 | 432 ms |
| 200 | 2,629 | 1,608 ms |
| 600 | 7,828 | 4,495 ms |
| 700 | 9,127 | **HTTP 400** |

The checkpoint enforces `max_state_length=9000`. The 700-repeat request fails
with:

```
state[0] has 9127 tokens; limit is 9000. Shorten the input, increase the
configured limit, or explicitly set truncation='truncate'.
```

Peak VRAM at 7,828 state tokens: 5,945 MiB. Longer states are the main memory
driver, not candidate count.

## Queue, backpressure, and timeouts

**Queue full.** `JEVA_QUEUE_SIZE=2`, `JEVA_MAX_BATCH=1`, 8 concurrent clients for
5 s: 9 requests succeeded, 3,490 were rejected with **429**, client-side p50
7 ms for the rejections. A full queue fails fast instead of growing memory.

**Timeouts.** `JEVA_REQUEST_TIMEOUT=0.05`, 1 client: **36 × 504**, and 37 × 429.
The 504s return in ~50 ms, but the GPU work already started cannot be cancelled:
the worker keeps finishing it, the queue fills, and later requests get 429. Plan
timeouts around the longest request you actually want to serve, not the other
way round.

## Stability

90 s of `decide3` at concurrency 4:

| Metric | Before | After |
| :-- | --: | --: |
| Requests | 0 | 728 |
| Throughput | | 8.09 req/s |
| p50 / p95 / p99 | | 496.6 / 517.0 / 549.3 ms |
| Errors | | 0 |
| VRAM | 4,325 MiB | 4,389 MiB |
| State cache | 0 entries | 2,048 entries / 6 MiB (2,193 misses, evictions active) |
| Option cache | 0 entries | 8 entries / 72 KiB |

No growth pattern beyond the bounded caches. State-cache eviction works under
unique-state load: 2,193 misses against a 2,048-entry cap.

## Operational summary

- **Single request, warm options:** ~120–170 ms for one question, ~490 ms for
  three.
- **Sustained throughput:** ~21–24 req/s for one-question decisions, ~8 req/s
  for three-question decisions on this GPU.
- **Best throughput/latency trade-off:** keep batching on; it is worth ~2× for
  cheap requests and ~1.3× for heavy ones.
- **Concurrency is not free:** the worker is serialized. Above the saturation
  point, p99 climbs into seconds while throughput stays flat.
- **Priming is the difference** between 83 s and 150 ms for a 2,048-option
  bank.
- **Hard limits:** 9,000 state tokens, 8,192 option tokens per option, 32 queued
  requests, 5.9 GiB peak VRAM at the tested sizes.

## Caveats

- One laptop, one GPU, one run per configuration. Medians over 12–90 s phases,
  not a statistical study.
- Synthetic prompts and a synthetic policy bank. Real tickets vary in length and
  tokenization.
- Localhost only: no TLS, no reverse proxy, no multi-replica routing. Those add
  their own latency and capacity limits.
- 4-bit only. Accuracy versus bf16 has not been measured; latency numbers here
  are not a quality claim.
- The RTX 4060 shares power and thermal budget with the laptop; steady-state
  clocks under longer runs may be lower than these 12-second phases.

## Reproduce

```bash
# start the server (see server/README.md), then:
cd server
../.venv/bin/python scripts/loadtest.py \
  --url http://127.0.0.1:8079 --modes noul,decide3,rank8 \
  --concurrency 1,2,4,8,16 --duration 12 --out scripts/results/base.json

../.venv/bin/python scripts/limits.py \
  --url http://127.0.0.1:8079 --out scripts/results/limits.json
```
