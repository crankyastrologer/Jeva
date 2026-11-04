# jevad architecture

`jevad` wraps the Jeva library in an HTTP service. This file is the map; the
README covers usage, and `OPTIMIZATION.md` covers the performance work.

## Module layout

| Module | Responsibility |
| :-- | :-- |
| `jevad/app.py` | `create_app` only: settings, FastAPI instance, state, middleware, routers |
| `jevad/lifecycle.py` | model loading, engine construction, startup/shutdown lifespan |
| `jevad/config.py` | `Settings` dataclass and environment parsing |
| `jevad/deps.py` | FastAPI dependencies (`SettingsDep`, `EngineDep`, `ModelDep`, `CreatedDep`) |
| `jevad/middleware.py` | Bearer auth on `/v1/*` plus request timing/queue metrics |
| `jevad/errors.py` | OpenAI-shaped error envelope and exception handlers |
| `jevad/execution.py` | `Job`, `JobKind`, `ModelAdapter`, `execute_jobs`, `submit` |
| `jevad/runtime.py` | `InferenceEngine`: queue, micro-batching, single worker thread |
| `jevad/converters.py` | request schema (`ChoiceSpec` etc.) to Jeva `Choice`/`Noul`/`Score` |
| `jevad/openai_spec.py` | fence stripping, decision-spec parsing, message text, usage mapping |
| `jevad/openai_responses.py` | OpenAI chat/completion responses and SSE generators |
| `jevad/streaming.py` | SSE framing, chunking, headers |
| `jevad/metrics.py` | Prometheus text renderer and counters |
| `jevad/stub.py` | deterministic fake model for tests, smoke runs and Docker probes |
| `jevad/routes/` | HTTP surface only; no model calls beyond `submit()` |
| `jevad/schemas/` | Pydantic request/response models split by domain, re-exported from `__init__` |

## Packaging

Two distributions with a one-way dependency:

* the repository root builds `jeva-model` (the library and its CLI);
* `server/pyproject.toml` builds `jevad` and depends on `jeva-model` being
  installed first, because the library is not published on PyPI.

Install order is therefore library, then server:

```bash
pip install -r requirements.txt      # from the repo root, installs jeva-model
pip install -e "server[dev]"         # jevad plus test/lint extras
```

`server/pyproject.toml` is the single source of truth for server dependencies;
there are no `server/requirements*.txt` files. The Dockerfile installs the
library, then `pip install ./server`.

## Request lifecycle

```
client
  │  HTTP
  ▼
middleware.install_middleware        auth + inflight/latency metrics
  ▼
routes/*.py                          validate payload, build questions, create Job
  ▼
execution.submit                     maps engine errors to HTTP status
  ▼
runtime.InferenceEngine.submit       enqueue (future, job)
  ▼
runtime worker                       collect up to max_batch jobs within batch_window
  ▼
execution.execute_jobs               run in a thread; decide jobs go through system_one_many
  ▼
execution.ModelAdapter               Jeva API (with fallbacks for models without bulk/cache APIs)
  ▼
runtime worker                       resolve futures, record batch metrics
  ▼
routes/*.py + openai_responses.py    response model or SSE stream
```

Only `decide` jobs are batched, because they share the encoder. `rank`, cache
operations and priming are executed individually in the same worker, which keeps
them serialized with inference.

## Invariants

- Routes never touch the model directly; everything goes through a `Job`.
- The worker thread is the only place model methods run, so Jeva's `RLock` is
  never contended.
- `execute_jobs` writes results by original index; `strict=True` zips make
  misalignment a hard error.
- Error payloads always use the OpenAI envelope (`error.message/type/code`).
- Route paths, payloads, env vars and metric names are part of the public
  contract; `tests/golden/responses.json` locks them.

## Adding a route

1. Add request/response models under `jevad/schemas/` (and re-export in
   `schemas/__init__.py`).
2. If it needs the model, add a `JobKind` and a handler in
   `execution._HANDLERS` (or extend `ModelAdapter`).
3. Add the endpoint to the relevant `jevad/routes/*.py` using `EngineDep` and
   `SettingsDep`.
4. Extend `tests/golden_cases.py` and regenerate `tests/golden/responses.json`.

## Tests and tooling

```bash
cd server
../.venv/bin/python -m pytest       # unit, API, golden and engine tests
../.venv/bin/ruff check jevad tests scripts
../.venv/bin/ruff format jevad tests scripts
```

`tests/golden_cases.py` collects normalized responses for every route; the
snapshot in `tests/golden/responses.json` is the behavior-freeze check used
throughout the refactor. Performance scripts live in `scripts/` and are not
collected by pytest.
