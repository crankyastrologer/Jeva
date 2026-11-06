<div align="center">

# Jeva
### System One Models from First Principles

**Qi Li · Xinchao Wang**  
National University of Singapore

[Model](https://huggingface.co/LIQIIIII/Jeva-4B) · [Project page](https://liqiiiii.github.io/Jeva/) · [Evaluation](docs/evaluation.md) · [Fine-tuning](docs/finetuning.md)

<img src="site/assets/structural-properties-tagline.svg" width="570" alt="Null-aware · Permutation-invariant · Option-scalable · Cacheable">

</div>

<p align="center"><a href="site/assets/figure1.pdf"><img src="site/assets/figure1.png" width="1000" alt="Jeva architecture and four structural properties: null awareness, permutation invariance, option scalability, and reusable option memory"></a></p>

> **Fork notice.** This is a fork of [LiQiiiii/Jeva](https://github.com/LiQiiiii/Jeva).
> The Jeva model and paper are by Qi Li and Xinchao Wang (NUS).
>
> This fork adds **jevad**, an OpenAI-compatible inference server under
> [`server/`](server/README.md), plus optional 4-bit/8-bit quantization, batched
> scoring, state/option/set caching, and cache priming in `src/jeva/`. The
> measured 4-bit footprint is 3.6 GiB on an 8 GB GPU.
>
> Start with [`server/README.md`](server/README.md),
> [`server/ARCHITECTURE.md`](server/ARCHITECTURE.md), and
> [`server/OPTIMIZATION.md`](server/OPTIMIZATION.md).

## Why I think System One models matter

Hi everyone. System One models are one of the most recent innovations in AI. After working in a lot of settings, from startups to big corporations, I keep noticing the same thing: most AI use cases don't actually need free flowing, stochastic text. They need reproducible, predictable decisions from a narrow set of choices.

When you handle millions of decisions a day, you can't just rely on an AI generating text and extracting a decision out of it. The main asset is the intelligence, not the language generation.

First there was structured output. We got LLMs to generate verifiable JSON, and then extracted the decision from it. It works, but it has its own problems. One of the big ones is token inefficiency. JSON itself is token heavy. You are paying the LLM to write out the decision word by word instead of just pointing to the correct option.

Here come System One models. Instead of generating text like a normal LLM, they only output a decision. You give the model a state (the information it has to decide about) and a question (what decision it has to make), along with typed options, and it returns the choice it thinks fits best, or a probability per option, a truth probability, or an expected rubric score. No generated text, no JSON, no parsing step.

### The trade-off

Now, this approach has advantages and disadvantages. With a normal LLM, each token it outputs can actually change its answer about something. The model doesn't choose an answer up front and then write it out; it is assessing and answering as it writes. That iterative process is where a lot of its intelligence comes from. A System One model has to get it right in one shot, in a single forward pass. So you lose some of that intelligence, and that is why these models are not that suited for applications like mathematics and complex coding. But where they excel is making easy decisions from a set of choices, fast and cheap, and a lot of enterprise functions rely on exactly that. For eg, you want to understand what a user chat is about before redirecting them to an agent, or you want to understand if a text contains PII before it goes through the main LLM. These are System One problems.

### Why serving looks different

Now let's talk about latency, because that is where these models get interesting. In usual LLM decode, you prefill all the input tokens, and then the LLM outputs token by token. For each token it computes attention with the older ones and generates a new one, until the output is complete. That loop is mostly a memory bandwidth bottleneck, and there are ways to alleviate it a little, like speculative decoding or multi token output.

A System One request has no decode loop. It is prefill only, so there is no KV cache to page and nothing for speculative decoding to accelerate. Continuous batching, in the strict sense of adding new requests into a running token stream, does not really exist here. Batching still matters, but at the request level: many independent texts can share one forward pass. You batch rows, you bound padded tokens, and you cache aggressively.

### Why Jeva is a good fit for this

Jeva in particular is special. From an inference perspective, one of the main advantages of its design is that it computes the state and question in one prompt, and options separately, until the final calculation. They never share a sequence inside the encoder. The interaction happens later, in a small 13M parameter set attention head that produces candidate probabilities and a separate coverage estimate for "is the answer even present here?"

This lets us cache the pieces separately. Options survive question changes. Changing the question means re-encoding the combined state and question prompt, because they are the same forward pass. Priming a candidate bank once takes it off the critical path. Exact repeats become a head call plus cache lookups. Because the no decode path is cheap, we can also afford things a token server cannot: keep option memory in CPU RAM or on disk and pay a few milliseconds to fetch it, or share a bank across replicas. Pushing hidden vectors to an offsite cache is possible too, but it brings privacy and consistency questions. I think of it as a direction, not a default.

### Jeva as a judge

This is the part I am most excited about. A lot of "LLM as a judge" work is really evaluation work: rank two answers, score a response against a rubric, check whether a claim is supported by the context. Jeva fits that shape really well.

It scores a fixed set of candidates, so position bias is not an issue. Reordering the options permutes the probabilities instead of changing them. It gives an explicit coverage or null probability, so you can say "none of these options fit, escalate" instead of picking the least bad one. And it is cheap enough to run on every request, not just in offline evals.

For hallucination detection specifically, you can give it a claim and the retrieved evidence and ask a Noul question: is this claim supported by this evidence? Coverage lets you separate "the model is wrong" from "the evidence does not answer this". At high volume, that is a much better fit than asking a large model to write out a verdict.

But there is one real limitation, and it is the thing that holds these models back for evaluations: you cannot ask Jeva to explain itself. You get a probability or a label, not a chain of thought. There is no reasoning trace to inspect, no "here is why I chose B". If the model gets a judgement wrong, you are blind to the thought process that led there. You can debug around it by changing the options, adding more context, or checking that the question is well posed, and you can calibrate it on labelled data, but you cannot read its reasoning. For regulated workflows or anything that needs an auditable rationale, that is a serious constraint.

### What a serving stack for this class needs

1. A way to prime (pre-encode) state/question prompts, individual options, or whole option sets.
2. Stable cache keys so a decision can address those pieces independently.
3. Tiered storage for that memory: GPU, CPU, disk, and eventually remote.
4. An API that accepts flexible input: vary the state while holding the question and options fixed, or swap candidates while holding the state fixed.

This repository is my attempt at that stack. It is not a vLLM replacement. It is a small, typed decision server built around those four ideas.

## What this fork adds

- `server/` is **jevad**, an OpenAI-compatible inference server with `/v1/decide`, `/v1/rank`, `/v1/chat/completions`, `/v1/completions`, cache priming, SSE streaming, Bearer auth, Prometheus metrics, and a Dockerfile.
- Optional 4-bit/8-bit loading (`bitsandbytes`) for GPUs that cannot hold the 8.8 GB backbone. Measured on an RTX 4060 Laptop: 3.6 GiB at 4-bit.
- Batched state encoding, request-level micro-batching, a state-encoding cache, and instruction-independent option memory.
- An engineering log with every benchmark and dead end: [`server/OPTIMIZATION.md`](server/OPTIMIZATION.md).

### Quick start

```bash
# library (see Installation below for the full environment)
python -m pip install -r requirements.txt
jeva-download LIQIIIII/Jeva-4B --local-dir ./models/Jeva-4B

# server
pip install -e "server[dev]"
JEVA_MODEL=./models/Jeva-4B JEVA_QUANTIZE=4bit \
JEVA_OPTION_CHUNK_SIZE=4 JEVA_OPTION_BATCH_TOKENS=4096 \
uvicorn jevad.app:app --port 8000
```

Then open `http://localhost:8000/docs`, or prime a candidate bank:

```bash
curl -s localhost:8000/v1/cache/prime -H 'Content-Type: application/json' \
  -d '{"candidates": {"refund": "Duplicate charges.", "delivery": "Shipment tracking."}}'
```

- Server guide: [`server/README.md`](server/README.md)
- Architecture: [`server/ARCHITECTURE.md`](server/ARCHITECTURE.md)
- Benchmarks and rationale: [`server/OPTIMIZATION.md`](server/OPTIMIZATION.md)

## Choices I made and why

Building this was mostly a series of small trade-offs. Here are the ones that mattered.

**Two packages, not one.** `jevad` is the server, `jeva-model` is the library. The library does the math, the server does HTTP, queues and caching. Keeping them separate makes each easier to reason about, though it does mean installing the library first.

**A native API, plus a chat compatibility layer.** `/v1/decide` is the interface I actually wanted: state, questions, probabilities. Most tools only speak chat completions, so that endpoint accepts a JSON spec in the last message and returns the decision as text. It trades some elegance for compatibility, and the docs are explicit about that.

**Why not vLLM.** The generation optimizations exist to make a decode loop faster. There is no decode loop here, so token-level scheduling doesn't really apply. A small queue that batches whole requests covers the actual workload without pulling in a full generation engine.

**One worker, short queue.** One GPU, one model lock. One worker takes jobs, waits 5ms, runs up to 8 decide requests in one pass. The queue is capped: 429 when full, 504 on timeout. Set max batch to 1 when latency matters more than throughput.

**Only decide gets batched.** Rank is already one call, and cache operations are cheap. Keeping them simple avoids unnecessary complexity.

**Caching turned out to matter most.** Options don't know about the state or question, so I can cache them by text, reuse them across questions, and prime a bank before traffic. Repeats went from ~130ms to ~10ms, and primed options cost zero. One caveat: caches are keyed by text, so they need to be per tenant. That's the next step.

**4-bit backbone, full precision head, no LoRA merge.** 8.8GB doesn't fit in 8GB. NF4 gets it to 3.6GiB. Merging into a quantized base isn't supported, so the adapter runs at runtime. I haven't compared 4-bit against bf16 accuracy, so it's worth validating on your own data.

**Streaming is chunked, not token-by-token.** The model emits no tokens. I split the decision JSON into chunks so chat clients work, and the docs describe it accurately.

**Freeze, then refactor.** Before splitting the 495 line app, I saved golden responses for every route and made them tests. The refactor is seven commits, tests green after each one.

**Still on the list:** tenant cache isolation, rate limits, SSO, audit logs, a batch API, and calibration. Those are the enterprise gaps I'd close next.

---

**Upstream documentation.** The sections below are the original Jeva documentation, kept for installation, inference, evaluation, and fine-tuning.

Jeva is a system one model (e.g., Jev) with native design. Give it a state and typed
questions; get candidate probabilities, a Boolean probability, or an expected
rubric score. The language encoder represents the state and each option
independently. A compact decision stack updates only the state against the
complete option memory, so decisions remain set-dependent while option
computation is reusable.

- **Permutation symmetry:** reordering candidates permutes their probabilities,
  rather than changing their semantic identities.
- **Variable candidate sets:** shared scoring does not impose a fixed label
  vocabulary. Encoding runs in bounded batches; all options reach the head.
- **Null awareness:** an explicit coverage estimate separates answer availability
  from competition between available candidates.
- **Two-level reuse:** cache complete option tuples, or reuse individual option
  vectors and projected keys/values across newly assembled sets.

## Citation

```bibtex
@article{li2026jeva,
  title={Jeva: System One Models from First Principles},
  author={Li, Qi and Wang, Xinchao},
  journal={arXiv preprint arXiv:2610.18773},
  year={2026}
}
```

## Installation

Use Linux x86_64, an NVIDIA GPU, and a clean conda environment:

```bash
conda create -n jeva -c nvidia/label/cuda-12.8.1 python=3.12 cuda-nvcc=12.8.93 -y
conda activate jeva
git clone https://github.com/LiQiiiii/Jeva.git
cd Jeva
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jeva-doctor --check-kernels --strict
```

The requirements pair PyTorch 2.9.1 / CUDA 12.8 with matching accelerated kernels.
A compatible NVIDIA driver is required. The first request may compile kernels.
See [runtime configuration](docs/runtime.md) for memory limits and the slower
reference installation in `requirements-core.txt`.

## Local inference

```python
from jeva import Jeva, Choice, Noul, Score

model = Jeva.from_pretrained("LIQIIIII/Jeva-4B", device="cuda")

r = model.system_one(
    state=(
        "The production login page returns HTTP 503. "
        "All customers are unable to sign in or complete purchases."
    ),
    questions={
        "outage": Noul(
            instructions="Does this report an ongoing service outage?"
        ),
        "team": Choice(
            instructions="Which team should handle this support ticket?",
            criteria={
                "billing": "Payment disputes, invoices, and refunds.",
                "engineering": "Service outages, software errors, and broken features.",
                "general": "All other customer support requests.",
            },
            allow_null=False,
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

print("Outage probability:", r.answers["outage"].noul)
print("Assigned team:", r.answers["team"].choice)
print("Team probabilities:", r.answers["team"].probabilities)
print("Impact score (0–2):", r.answers["impact"].score)

```

| Interface | Result |
| :-- | :-- |
| `Choice` | A candidate key and its probability distribution |
| `Choice(..., allow_null=True)` | A candidate key or `None`, with explicit null probability |
| `Noul` | Probability that the statement is true, between 0 and 1 |
| `Score` | Expected rubric level; supplied `values` define custom numerical units |

For open-set Choice, candidate probabilities plus `null_probability` sum to one.
`conditional_probabilities` reports preference conditional on selecting a candidate.
Noul is a truth probability, not the null outcome. Exact ties use the first
candidate; open-set ties place null first.

Mappings format options as `label: description`; lists preserve their text.
For exact dataset option text and order, use `model.predict(...)`.
The [runnable example](examples/local_inference.py) returns all three output types.


### Cache and reuse

```python

policies = {
    "refund": (
        "Duplicate payments and refunds. When a customer reports being charged "
        "twice for the same purchase, first verify that both transactions have "
        "settled and refer to the same order. A pending authorization is not a "
        "second completed payment. If two settled charges are confirmed, refund "
        "the duplicate transaction to the original payment method and retain "
        "the valid payment. Record both transaction identifiers in the support "
        "case. Tell the customer that the refund normally appears within five "
        "to ten business days. This procedure does not cancel the order or "
        "change its delivery status."
    ),
    "cancellation": (
        "Order cancellation before dispatch. Customers may cancel an order "
        "while its fulfillment status is unshipped. Check the warehouse record "
        "before confirming cancellation. If the package has not been handed "
        "to the carrier, stop fulfillment, release reserved inventory, and "
        "refund the original payment. Send a cancellation confirmation after "
        "the warehouse acknowledges the request. If the package has already "
        "been dispatched, do not promise that shipment can be stopped; explain "
        "the return procedure instead. This policy concerns cancellation of "
        "the entire purchase, rather than correction of a duplicate charge."
    ),
    "delivery": (
        "Shipment tracking and delivery delays. For an order that has already "
        "been dispatched, retrieve the carrier name, tracking number, latest "
        "scan, and estimated delivery date. Explain the current shipment status "
        "without guaranteeing a delivery time not confirmed by the carrier. "
        "If tracking has not updated for three business days, open a carrier "
        "investigation and notify the customer when new information arrives. "
        "Check the delivery address if the carrier reports an address problem. "
        "Do not cancel a dispatched order through the warehouse cancellation "
        "process or issue a duplicate-payment refund."
    ),
    "account": (
        "Customer contact information and account updates. After verifying "
        "the customer's identity, support staff may update the email address "
        "or telephone number associated with an account. Confirm the new "
        "information and send an account-change notification. Never request "
        "a password or a full payment-card number in a support conversation. "
        "Changing account contact details does not automatically modify the "
        "delivery address of an existing order. For an order already in "
        "fulfillment, check its status and follow the relevant address-change "
        "procedure. This policy does not authorize refunds or cancellations."
    ),
}

model.cache = "hybrid"
model.clear_cache()

r1 = model.rank(
    state="I was charged twice for the same order. How can I get one charge refunded?",
    candidates=policies,
)

r2 = model.rank(
    state="My package has shipped, but tracking has not updated for four business days.",
    candidates=policies,
)

request = "Please cancel my order. The warehouse confirms it has not shipped."
subset = {key: policies[key] for key in ["refund", "cancellation", "account"]}

r3 = model.rank(state=request, candidates=subset)

model.cache = "none"
try:
    r3_off = model.rank(state=request, candidates=subset)
finally:
    model.cache = "hybrid"

print(
    f"\n{'Request':<25}"
    f"{'Top choice':<15}"
    f"{'Latency (ms)':>14}"
    f"{'Encoded tokens':>17}"
    f"{'Cached tokens':>16}"
)

for name, result in [
    ("1. Cold policy bank", r1),
    ("2. Same policy bank", r2),
    ("3. New candidate subset", r3),
    ("3. Same request, Off", r3_off),
]:
    print(
        f"{name:<25}"
        f"{result.ranking[0].label:<15}"
        f"{result.latency_ms:>14.1f}"
        f"{result.usage.encoded_option_tokens:>17}"
        f"{result.usage.cached_option_tokens:>16}"
    )

print(f"\nSame-request Off / reuse latency: {r3_off.latency_ms / r3.latency_ms:.2f}x")
```
With hybrid caching enabled, Jeva reuses option representations across changing requests and newly assembled candidate sets, while computing each decision anew.


### Offline loading

```bash
jeva-download LIQIIIII/Jeva-4B --local-dir ./models/Jeva-4B
```

```python
model = Jeva.from_pretrained("./models/Jeva-4B", local_files_only=True)
```

## Evaluation

Run the model on a local JSONL file:

```bash
jeva-evaluate \
  --model ./models/Jeva-4B \
  --data ./my_eval.jsonl \
  --output ./results/my_eval
```

The [evaluation guide](docs/evaluation.md) covers input format, metrics,
permutations, cache comparisons, and resumable runs. 

<p align="center"><a href="site/assets/structural_properties.pdf"><img src="site/assets/structural_properties.png" width="1200" alt="Four experiments from the paper: null detection, option-order stability, candidate scaling, and cache reuse"></a></p>

Jeva achieves 78.30% Typed Decisions accuracy, 98.66% null-detection
AUROC, and zero observed flips on 600 Choice items. Jeva handles 2,048 candidates
at 69.88% accuracy and 12.04 s median latency. Full option reuse across newly
assembled MLDR sets yields a 27.98× ratio of median request latencies, excluding
separately measured priming. 

## Fine-tune on your own data

```bash
jeva-finetune \
  --model ./models/Jeva-4B \
  --train-data ./my_train.jsonl \
  --validation-data ./my_validation.jsonl \
  --output-dir ./runs/my-jeva \
  --mode head --device cuda \
  --epochs 3 --batch-size 1 --gradient-accumulation 8
```

`head` trains the decision module and projections while freezing the encoder.
`lora` also updates the existing adapter. Load the resulting directory with
`Jeva.from_pretrained()`.

See [fine-tuning](docs/finetuning.md) for the data schema, objectives, validation,
and resume. The small [JSONL example](examples/finetune_data.jsonl) illustrates
the format. Install `pip install -e '.[train]'`
only if you want to read your training data through the Hugging Face datasets API.

## Repository

```text
src/jeva/      local API, model runtime, evaluation, fine-tuning, installation checks
examples/      inference example and training-format illustration
train/         fine-tuning command wrapper
docs/          installation, evaluation, and fine-tuning guides
site/          project page, paper figures, and interactive diagrams
.github/       project-page deployment workflow
```



##  License
Code: [Apache-2.0](LICENSE). Model weights and datasets retain their respective
licenses.
