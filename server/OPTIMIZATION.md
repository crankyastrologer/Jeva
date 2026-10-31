# Jeva performance work — engineering log

This document records every change made to get Jeva-4B running on an 8 GB
laptop GPU and to make the inference server faster, with the reason for each
change and the measurements behind it.

Machine: Windows 11 + WSL2 Ubuntu 24.04 (kernel 6.6.87.2), RTX 4060 Laptop
(8188 MiB VRAM, sm_89), 7 GB WSL RAM, Python 3.12.3, PyTorch 2.9.1+cu128,
Jeva checkpoint `LIQIIIII/Jeva-4B` (backbone `qwen3_5`, LoRA adapter, decision
head).

---

## 1. Baseline and constraints

Measured before the performance work, with the real checkpoint on this GPU
(4-bit, served over HTTP):

| Phase | Result |
| :-- | :-- |
| 3-question request, first call after startup | 8.8 s total (7.98 s + 0.45 s + 0.39 s per question) |
| Steady-state 3-question request, cold state | ~450 ms |
| BF16 load | impossible: 8.8 GB backbone vs 8188 MiB VRAM and 7 GB WSL RAM |
| 4-bit load | 3.6 GiB allocated, 4.05 GiB peak, ~52–90 s one-time |

Two structural facts drove the design:

1. **Jeva is an encoder, not a generator.** Each question is
   `pool(last token) -> projection -> decision stack`. There is no decode loop,
   so paged attention / prefix-KV tricks for generation do not apply.
2. **The state is the *suffix* of the prompt.**
   `state_question_text` = `"Decision type: {type}\nQuestion: {instructions}\nState:\n{state}"`.
   An early idea to reuse one state encoding across questions by sharing a
   prefix is impossible without changing the trained prompt format. This is
   recorded as a correction to the initial optimization proposal.

So the available wins were:

* fewer encoder forwards per request (batch the rows the model already accepts),
* reuse across repeated requests (state/option/set caches),
* batching across concurrent requests,
* and not doing things that only help token generation.

---

## 2. Changes

### 2.1 4-bit loading (prerequisite, previously added)

Files: `src/jeva/_runtime/backbone.py`, `src/jeva/_runtime/model.py`,
`src/jeva/inference.py`, `server/jevad/config.py`, `server/jevad/app.py`.

* `TextBackbone(..., quantize=..., device=...)` builds a `BitsAndBytesConfig`
  (NF4, bf16 compute, double quant) and loads with `device_map` set to the GPU
  index. The backbone is what does not fit, so only it is quantized.
* `System1DecisionModel` forwards the flag, and moves the small head modules
  (`context_proj`, `option_proj`, `core`) to the GPU explicitly, because a
  quantized model cannot be moved with `.to()` afterwards.
* `Jeva.from_pretrained(..., quantize="4bit")` requires `device="cuda"`, skips
  the LoRA merge (merging into a quantized base is unsupported) and skips the
  post-load `.to(device)`.
* Rationale: 8.8 GB of bf16 weights cannot fit in 8 GB VRAM; 4-bit needs
  ~2.2 GB of weights and leaves room for activations. The LoRA adapter (496 MB)
  stays separate and is applied by PEFT at runtime.

### 2.2 Batched encoding inside one request

File: `src/jeva/inference.py`.

Previously every question in `system_one` went through `_predict`, which
tokenized and encoded its own state row (one 4B forward per question). The new
flow prepares all questions first (`_prepare`), then:

* encodes **all state rows in one batched forward** (`_encode_state_rows`,
  chunked by `state_chunk_size=8` and `state_batch_tokens=32768`),
* resolves options once for the whole call and assembles each question's option
  memory,
* runs the tiny decision stack per question.

Rationale: the encoder was called 3× for a 3-question request with 40-token
prompts; one forward with batch 3 is far better utilization. This is the single
largest single-request win measured (2.0–2.7×).

### 2.3 Cross-request micro-batching

Files: `server/jevad/runtime.py`, `server/jevad/app.py`.

* `InferenceEngine` now accumulates queued jobs for `batch_window` seconds
  (default 5 ms) or up to `max_batch` (default 8), then executes them with a
  single executor call in a worker thread.
* The executor partitions jobs: all `decide` jobs go through the new
  `Jeva.system_one_many(requests)`, which concatenates their questions and runs
  one `_decide_batch`; `rank`, `cache_info`, `cache_clear` are handled inline.
* `system_one_many` returns one `DecisionResponse` per request, with per-request
  usage and latency.
* Rationale: the model already accepted a list of rows; the old server
  serialized whole requests and threw that capacity away. Batching is the only
  way to use the GPU under concurrency without vLLM.

### 2.4 State-encoding cache

File: `src/jeva/inference.py`.

* New `BoundedCache` for state encodings (`state_cache_size=2048`,
  `state_cache_max_bytes=128 MiB`), keyed by the token ids of the state prompt.
* `clear_cache()` clears it and `/v1/cache` reports a third `states` section.
* Rationale: repeated identical requests (retries, dashboards, evaluations that
  replay the same item) previously re-encoded the 4B state every time. With the
  cache, the whole request path becomes option-memory reuse plus a 768-dim
  decision stack.

Honest limitation: because the state is at the end of the prompt, two different
questions on the same state produce different token ids and cannot share this
cache entry. The cache only helps exact repeats.

### 2.5 Option deduplication across requests

File: `src/jeva/inference.py` (`_decide_batch`).

* With `cache="hybrid"`, option rows are collected across *all* requests in the
  micro-batch, looked up in the per-option cache, and the missing ones are
  encoded once, in one batched forward, then projected into per-option memories.
* Usage accounting reports per request how many option tokens were encoded and
  how many were cache hits.
* Rationale: concurrent requests usually share a candidate bank (support
  routing, policy selection). Before this, a cold batch of N requests encoded
  the same options N times.

### 2.6 Explicit option-bank priming

Files: `src/jeva/inference.py`, `server/jevad/{schemas,app,stub}.py`.

Jeva separates option memory from the question prompt: `option_text(option,
decision_type)` depends only on the option text and the decision type, never on
the instructions or state. The cache keys are therefore stable across question
wording. What was missing was a way to populate that memory before traffic, so
the first real request still paid the cold encode.

Added:

* `Jeva.prime_options(options, kind=...)`, `Jeva.prime_question(question)`, and
  `Jeva.prime(questions)` in the library. They tokenize and encode the unique
  option rows once, store per-option memories, and assemble/store the exact set
  memory so the next decision is a set-cache hit. With `cache="set"` they encode
  and store the set directly. Returns token/caching stats.
* `POST /v1/cache/prime` with either `{"questions": <decide schema>}` or
  `{"candidates": {...}|[...], "kind": "choice"}`. It runs as a `prime` job
  through the same engine, so it serializes with inference but never touches the
  state encoder.
* Mapping candidates are converted to `label: description` texts, matching what
  `/v1/rank` and `/v1/decide` feed the model.

Measured with an 8-policy bank (8 options, 185 option tokens, 4-bit):

| Scenario | Median per request | Option tokens encoded |
| :-- | :-- | :-- |
| Cold, no priming | 368 ms (first call 1090 ms) | 185 on the first call, 0 after |
| After `prime_options` (310 ms one-time) | 140 ms (first call 148 ms) | 0 |
| `cache="none"` | 429 ms | 185 every call |
| Primed, **instructions changed** (3 variants) | 109–114 ms | 0 |

The last row is the point of the feature: changing the question does not
invalidate option memory, so a routing/triage service can prime its candidate
bank once and then vary instructions and states freely. Priming also removes the
first-request cold spike (1090 ms → 148 ms), which matters for autoscaled
deployments that cannot afford a slow first user.

### 2.7 Optional `torch.compile` of the decision stack

Files: `src/jeva/inference.py`, `server/jevad/config.py`.

* `compile_core=True` wraps `model.decide_from_option_cache`; the wrapper falls
  back to eager on the first failure.
* **Measured slower** on this GPU (370 ms median vs 262 ms eager, plus a 17 s
  compile warm-up), because the stack is three small 768-dim matmuls and
  Inductor guard overhead dominates. It is therefore **off by default** and
  documented as not recommended for this class of GPU. Kept as an opt-in flag
  for larger GPUs where kernels dominate.

### 2.8 Server plumbing

Files: `server/jevad/{runtime,metrics,config,schemas,stub}.py`,
`server/scripts/*`.

* Metrics: `jeva_batches_total`, `jeva_batch_jobs_total` added; queue depth and
  inflight gauges already existed.
* Config: `JEVA_BATCH_WINDOW_MS`, `JEVA_MAX_BATCH`, `JEVA_COMPILE`; the model
  path auto-detects `./models/Jeva-4B` so the server no longer re-downloads from
  the Hub when a local checkpoint is present.
* Schemas: `CacheInfoOut.states`, `PrimeRequest` (exactly one of `questions` or
  `candidates`).
* Stub model implements `system_one_many` and records batch sizes, so batching
  can be smoke-tested without weights.
* Tests grew from 17 to 23 (library batching/caching tests plus engine batching).

---

## 3. Correctness and numerics

Batched encoder passes must not change decisions. Checked on the reference
3-question request (4-bit):

| Path | outage (Noul) | team (Choice) | impact (Score) |
| :-- | :-- | :-- | :-- |
| one-state-per-forward (`state_chunk_size=1`) | 0.925394 | engineering 0.661984 | 1.967081 |
| batched states (`state_chunk_size=8`) | 0.925729 | engineering 0.659746 | 1.967070 |
| cross-request batch (`system_one_many`) | 0.926803 | engineering 0.665848 | 1.966503 |

Decisions and scores are stable; probabilities differ at ~1e-3, which matches
the repository's own note that different batch shapes and kernel reductions
introduce floating-point differences. The earlier single-call value (0.926992)
is inside that band.

State text sensitivity is real and unrelated to batching: appending `" Warmup."`
to the state moved the ticket from `engineering` to `general` and outage from
0.926 to 0.827. Avoid mutating state text for warm-up in production; warm up
with a real request instead.

---

## 4. Benchmarks

All numbers on the RTX 4060 with `quantize="4bit"`, `option_chunk_size=4`,
`option_batch_tokens=4096`.

### 4.1 Within-request batching (A/B, two passes)

| state rows per forward | median latency (3-question request) |
| :-- | :-- |
| 1 (old behavior) | 360.6 ms / 324.2 ms |
| 3 (batched, new) | 122.0 ms / 160.3 ms |

**2.0–2.7× faster**, cold state, warm option cache.

### 4.2 Cache reuse

| Scenario | Median |
| :-- | :-- |
| New state, warm option cache | 261.8 ms (range 244–995) |
| Exact repeated request (state+option+set hits) | 10.1 ms (range 8.9–584) |

The first repeat still pays one cold-path outlier; steady state is ~10 ms
(≈26× faster). Cache counters after the repeats: 6 state hits, 16 option hits,
6 set hits.

### 4.3 Cross-request batching, library level

| Batch size | First pass (cold states) | Steady pass | Steady per request |
| :-- | :-- | :-- | :-- |
| 1 | 191 ms | 7 ms | 7 ms |
| 2 | 228 ms | 13–23 ms | 6 ms |
| 4 | 460 ms | 21–22 ms | 5 ms |
| 8 | 781 ms | 41–44 ms | 5 ms |

Cold throughput roughly doubles from 1 to 8 requests; steady state is bounded
by per-request decision overhead because the encoder work is already cached.

### 4.4 HTTP end-to-end, batching off vs on

Four concurrent clients, distinct states, warm options; two rounds each.

| Config | Round | 4 sequential | 4 concurrent | Gain |
| :-- | :-- | :-- | :-- | :-- |
| `JEVA_MAX_BATCH=1`, window 0 | 0 | 492 ms | 575 ms | 0.86× |
| `JEVA_MAX_BATCH=1`, window 0 | 1 | 526 ms | 507 ms | 1.04× |
| `JEVA_MAX_BATCH=8`, window 5 ms | 0 | 558 ms | 541 ms | 1.03× |
| `JEVA_MAX_BATCH=8`, window 5 ms | 1 | 544 ms | 439 ms | 1.24× |

Steady-state sequential median per 3-question request: 122–137 ms. Micro-batching
gives ~1.2× on four concurrent requests; the ceiling is low because each request
is small and 4-bit dequantization dominates. It costs a fixed 5 ms of queuing
latency, so `JEVA_MAX_BATCH=1` is the right choice for latency-sensitive,
low-concurrency use.

### 4.5 Compiled decision stack

| Path | Median | First call |
| :-- | :-- | :-- |
| eager | 261.8 ms | 12.8 s warm-up |
| `torch.compile` | 370.2 ms | 17.4 s compile + warm-up |

**1.4× slower.** Disabled by default.

---

## 5. Configuration and API summary

| Setting | Default | Purpose |
| :-- | :-- | :-- |
| `state_chunk_size` / `state_batch_tokens` | 8 / 32768 | state-row batching bounds |
| `state_cache_size` / `state_cache_max_bytes` | 2048 / 128 MiB | repeated-request cache |
| `compile_core` | `False` | opt-in `torch.compile` |
| `JEVA_BATCH_WINDOW_MS` | 5 | micro-batch wait |
| `JEVA_MAX_BATCH` | 8 | micro-batch size (`1` disables) |
| `JEVA_COMPILE` | `false` | same as `compile_core` |

Library API addition: `Jeva.system_one_many([(state, questions), ...])` returns
one `DecisionResponse` per input. `Jeva.cache_info()` now includes `states`.
`Jeva.prime_options(options, kind=...)`, `Jeva.prime_question(question)`, and
`Jeva.prime(questions)` pre-encode options and sets; the server exposes them as
`POST /v1/cache/prime`.

---

## 6. Tests

`server/tests` (pytest):

* `test_api.py` — endpoints, errors, auth, SSE, metrics, engine batching, priming.
* `test_library_batch.py` — state/option cache reuse, cross-request option
  dedupe, per-question state rows, `cache="none"` behavior, named/list questions,
  prime-then-decide reuse across changed instructions.

Result: **28 passed**. Performance scripts (not part of CI):

* `bench_real.py`, `bench_shapes.py`, `bench_ab.py` — in-process measurements.
* `bench_prime.py` — cold vs primed vs `cache="none"`, and instruction changes.
* `check_prime_http.py` — primes over HTTP, then verifies `encoded_option_tokens
  == 0` for changed instructions.
* `bench_http.py` — end-to-end concurrency measurement against a running server.
* `check_numerics.py` — batch-shape parity check.
* `real_check.py` — real checkpoint load, VRAM and latency sanity check.

---

## 7. Why not vLLM

vLLM's headline features solve decode-time problems. Jeva has no decode loop, so
paged attention, prefix caching for generations, and speculative decoding do not
apply. What vLLM would genuinely provide is its Qwen3.5 kernels (the checkpoint's
backbone is `qwen3_5`, which vLLM now lists natively, including the hybrid linear
attention layers) and batched prefill.

Options and effort remain roughly as assessed earlier:

* vLLM as a pooling/embedding backend plus a Jeva head sidecar: 1–2 weeks,
  needs exact parity for tokenization, last-token pooling, and the LoRA adapter,
  plus inter-process overhead.
* Full vLLM plugin with the head inside the worker and a custom `/decide` route:
  4–6 weeks.
* Upstream-grade "Jeva task" in vLLM: months, with numerical parity risk because
  the HF path uses FLA kernels and vLLM uses its own linear-attention kernels.

Given the measured 122–137 ms per 3-question request and ~10 ms for repeats,
vLLM is not justified for this deployment. The remaining wins are in caching and
batching, which are now in place.

---

## 8. Limitations and next steps

* The state cache cannot be shared across questions (state is the prompt
  suffix). Changing the prompt format to put the state first would enable
  prefix-KV reuse but invalidates the trained format; do not do this casually.
* Micro-batching is orthogonal to per-request latency; keep `max_batch=1` for
  interactive use, raise it for batch jobs.
* The decision stack is small; further gains are more likely from
  quantization-aware kernels for the 24 linear-attention layers than from
  compiling the head.
* Streaming is chunked JSON, not token streaming, by design.
* One model process per 8 GB GPU: loading a second 4-bit instance alongside the
  server exhausted VRAM and crashed the WSL service (`Wsl/Service/E_UNEXPECTED`).
  Recovery was `wsl --shutdown`, then restart. Benchmarks were rerun after that
  and are the numbers above.

---

## 9. Reproduction

```bash
cd ~/Jeva && source .venv/bin/activate

# full test suite
cd server && ../.venv/bin/python -m pytest && cd ..

# real-model sanity + VRAM
python server/scripts/real_check.py

# cache/batching benchmarks (one model process at a time)
python server/scripts/bench_real.py
python server/scripts/bench_shapes.py
python server/scripts/bench_ab.py
python server/scripts/bench_prime.py
python server/scripts/check_numerics.py

# HTTP A/B (batching on)
JEVA_MODEL=./models/Jeva-4B JEVA_QUANTIZE=4bit \
JEVA_OPTION_CHUNK_SIZE=4 JEVA_OPTION_BATCH_TOKENS=4096 \
JEVA_MAX_BATCH=8 JEVA_BATCH_WINDOW_MS=5 \
uvicorn jevad.app:app --app-dir server --port 8079
python server/scripts/bench_http.py
```
