# jevad — inference server for Jeva

FastAPI server exposing Jeva typed decisions over an OpenAI-compatible surface.

Jeva is not a chat model: it scores candidate options you supply. The
`/v1/chat/completions` and `/v1/completions` endpoints therefore expect the
request content to be a JSON decision spec and return the decision as JSON text.

## Endpoints

| Method | Path | Purpose |
| :-- | :-- | :-- |
| POST | `/v1/decide` | Native: state + typed questions -> probabilities |
| POST | `/v1/rank` | Native: state + candidate bank -> ranking |
| POST | `/v1/chat/completions` | OpenAI shim (JSON spec in last user message) |
| POST | `/v1/completions` | OpenAI shim (JSON spec as prompt) |
| GET | `/v1/models`, `/v1/models/{id}` | Model listing |
| GET | `/v1/cache` | Option/set/state cache statistics |
| POST | `/v1/cache/prime` | Encode a candidate bank or question options without deciding |
| POST | `/v1/cache/clear` | Clear caches |
| GET | `/healthz` | Liveness/readiness |
| GET | `/metrics` | Prometheus text metrics |
| GET | `/docs` | OpenAPI UI |

Requests to `/v1/*` require `Authorization: Bearer $JEVA_API_KEY` when
`JEVA_API_KEY` is set. `/healthz` and `/metrics` stay open.

## Quick start

```bash
cd ~/Jeva
source .venv/bin/activate
pip install -r server/requirements.txt
export JEVA_MODEL=./models/Jeva-4B
uvicorn jevad.app:app --app-dir server --host 0.0.0.0 --port 8000
```

On an 8 GB GPU use NF4 quantization and smaller option batches:

```bash
JEVA_MODEL=./models/Jeva-4B JEVA_QUANTIZE=4bit \
JEVA_OPTION_CHUNK_SIZE=4 JEVA_OPTION_BATCH_TOKENS=4096 \
uvicorn jevad.app:app --app-dir server --port 8000
```

No weights yet? Run the stub (fixed fake scores, no GPU):

```bash
uvicorn jevad.stub:app --app-dir server --port 8000
```

## Decision spec

```json
{
  "state": "The production login page returns HTTP 503 for all customers.",
  "questions": {
    "outage": {"type": "noul", "instructions": "Does this report an ongoing outage?"},
    "team": {
      "type": "choice",
      "instructions": "Which team should handle this ticket?",
      "criteria": {"billing": "Payments and refunds.", "engineering": "Outages and errors."},
      "allow_null": false
    },
    "impact": {
      "type": "score",
      "instructions": "Rate the customer impact.",
      "criteria": ["None", "Some", "All"],
      "values": [0, 1, 2]
    }
  }
}
```

## Usage

```bash
curl -s localhost:8000/v1/decide \
  -H 'Content-Type: application/json' \
  -d @spec.json | jq

curl -sN localhost:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"LiQiiiii/Jeva-4B\",\"stream\":true,\"messages\":[{\"role\":\"user\",\"content\":$(jq -Rs . spec.json)}]}"
```

Python with the OpenAI SDK (no API key needed when auth is disabled):

```python
import json
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="unused")
spec = json.load(open("spec.json"))

reply = client.chat.completions.create(
    model="LiQiiiii/Jeva-4B",
    messages=[{"role": "user", "content": json.dumps(spec)}],
)
print(json.loads(reply.choices[0].message.content))
```

Ranking:

```bash
curl -s localhost:8000/v1/rank -H 'Content-Type: application/json' -d '{
  "state": "I was charged twice for the same order.",
  "candidates": {"refund": "Duplicate charge handling.", "cancel": "Cancel an unshipped order."},
  "top_k": 1
}'
```

## Cache priming

Option memory is keyed by the option text and decision type only — instructions
and state are never part of the key. A candidate bank can therefore be encoded
once, before traffic, and reused by every later request even when the question
text changes.

```bash
curl -s localhost:8000/v1/cache/prime -H 'Content-Type: application/json' -d '{
  "candidates": {
    "refund": "Duplicate payments and refunds.",
    "cancellation": "Cancel an order before dispatch.",
    "delivery": "Shipment tracking and delays."
  }
}'
```

`candidates` may be a mapping (`label: description`, as used by `/v1/rank`) or a
list of option texts. Alternatively pass `questions` with the same schema as
`/v1/decide` to prime a complete typed question set; `kind` selects the option
prefix (`choice` by default).

Response:

```json
{
  "primed": {
    "kind": "choice", "options": 3, "encoded_options": 3,
    "encoded_tokens": 48, "cached_tokens": 0,
    "truncated_tokens": 0, "set_cached": true, "latency_ms": 96.4
  },
  "cache": {"mode": "hybrid", "options": {"entries": 3}, "sets": {"entries": 1}, "states": {"entries": 0}}
}
```

On an RTX 4060 with the 4-bit checkpoint, priming an 8-policy bank
(185 option tokens) costs ~310 ms once. Afterwards every request reports
`encoded_option_tokens: 0`, including requests that change the instructions.

## Configuration

| Variable | Default | Meaning |
| :-- | :-- | :-- |
| `JEVA_MODEL` | `LIQIIIII/Jeva-4B`, or `./models/Jeva-4B` if present | Hub id or local checkpoint directory |
| `JEVA_MODEL_NAME` | model id | Name reported by `/v1/models` |
| `JEVA_DEVICE` | `cuda` | `cuda` or `cpu` |
| `JEVA_CACHE` | `hybrid` | `none`, `set`, or `hybrid` |
| `JEVA_API_KEY` | unset | Enables Bearer auth on `/v1/*` |
| `JEVA_QUEUE_SIZE` | `32` | Pending request capacity (429 when full) |
| `JEVA_REQUEST_TIMEOUT` | `180` | Seconds before a queued request returns 504 |
| `JEVA_OPTION_CHUNK_SIZE` | `16` | Options encoded per sub-batch |
| `JEVA_OPTION_BATCH_TOKENS` | `16384` | Padded-token bound per sub-batch |
| `JEVA_MAX_STATE_LENGTH` | checkpoint | State token limit (truncation errors otherwise) |
| `JEVA_MAX_OPTION_LENGTH` | checkpoint | Per-option token limit |
| `JEVA_LOCAL_FILES_ONLY` | `false` | Disable Hugging Face network access |
| `JEVA_QUANTIZE` | unset | `4bit` or `8bit` (bitsandbytes, CUDA only); keeps LoRA unmerged |
| `JEVA_BATCH_WINDOW_MS` | `5` | Time to wait for more requests before running a micro-batch |
| `JEVA_MAX_BATCH` | `8` | Maximum requests per micro-batch (set `1` to disable batching) |
| `JEVA_COMPILE` | `false` | `torch.compile` the decision stack (slower on small GPUs) |
| `JEVA_HOST`, `JEVA_PORT` | `0.0.0.0`, `8000` | Bind address for `python -m jevad` |

On an 8 GB GPU reduce `JEVA_OPTION_CHUNK_SIZE` to `4` and
`JEVA_OPTION_BATCH_TOKENS` to `4096`.

## Docker

Build from the repository root (the image installs the library from source):

```bash
docker build -f server/Dockerfile -t jevad .
docker run --gpus all -p 8000:8000 \
  -v "$PWD/models/Jeva-4B:/models/Jeva-4B:ro" \
  -e JEVA_MODEL=/models/Jeva-4B \
  jevad
```

## Behaviour notes

- Concurrent requests are collected into a micro-batch (`JEVA_BATCH_WINDOW_MS`,
  `JEVA_MAX_BATCH`) and executed as one batched encoder pass; a single worker
  still serializes the decision step. Within one request, all question states
  are encoded in one forward.
- State encodings are cached by token ids alongside the existing option and set
  caches; `/v1/cache` reports all three. Repeating an identical request after a
  cold first call drops from ~130 ms to ~10 ms on an RTX 4060.
- Option memory is instruction-independent. Priming a candidate bank with
  `/v1/cache/prime` makes later requests option-cache hits even when the
  question wording changes.
- `usage.prompt_tokens` is Jeva `input_tokens` (state + newly encoded options);
  `completion_tokens` is 0. Encoded/cached token details are in `/v1/decide`.
- Streaming chunks are the serialized decision split every 256 characters; there
  is no token-level decoding because the model does not generate text.
- `temperature`, `top_p`, and `max_tokens` are accepted and ignored.
- `JEVA_COMPILE` is off by default: compiling the small decision stack measured
  slower than eager on this GPU (see `OPTIMIZATION.md`).

## Tests and linting

```bash
cd server
../.venv/bin/python -m pytest
../.venv/bin/ruff check jevad tests scripts
../.venv/bin/ruff format jevad tests scripts
```

The suite covers unit behavior, the HTTP API, a golden response snapshot of
every route, and engine batching. Performance scripts live in `scripts/` and are
not collected by pytest.

## Layout

`ARCHITECTURE.md` is the module map and request lifecycle. In short:
`app.py` only wires the app, `routes/` owns HTTP, `execution.py` owns jobs,
`runtime.py` owns the queue, `lifecycle.py` owns model startup, and `schemas/`
holds the Pydantic models.

