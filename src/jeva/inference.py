from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from threading import RLock
from time import perf_counter

import torch

from .types import Answer, Choice, DecisionResponse, Noul, RankResult, RankedItem, Score, Usage
from ._runtime.cache import BoundedCache
from ._runtime.collate import option_text, state_question_text, tokenize_texts
from ._runtime.config import Config, load_config


@dataclass
class _PreparedDecision:
    kind: str
    labels: list[str]
    options: list[str]
    allow_null: bool
    values: list[float] | None
    positive_index: int
    state_row: dict
    state_length: int
    truncated_state: int
    option_rows: list[dict]
    option_lengths: list[int]
    truncated_options: int
    keys: list[tuple]
    set_key: tuple


class Jeva:


    def __init__(self, model, config: Config, *, device="cuda", cache="hybrid",
                 max_state_length=None, max_option_length=None, truncation="error",
                 option_chunk_size=16, option_batch_tokens=16384,
                 option_cache_size=4096, option_cache_max_bytes=256 << 20,
                 set_cache_size=64, set_cache_max_bytes=256 << 20,
                 state_cache_size=2048, state_cache_max_bytes=128 << 20,
                 state_chunk_size=8, state_batch_tokens=32768,
                 quantize=None, compile_core=False):
        if cache not in {"none", "set", "hybrid"}:
            raise ValueError("cache must be 'none', 'set', or 'hybrid'")
        if truncation not in {"error", "truncate"}:
            raise ValueError("truncation must be 'error' or 'truncate'")
        if quantize not in {None, "4bit", "8bit"}:
            raise ValueError("quantize must be None, '4bit', or '8bit'")
        if any(type(v) is not int or v < 1 for v in (option_chunk_size, option_batch_tokens)):
            raise ValueError("option_chunk_size and option_batch_tokens must be positive")
        if any(type(v) is not int or v < 1 for v in (state_chunk_size, state_batch_tokens)):
            raise ValueError("state_chunk_size and state_batch_tokens must be positive")
        self.model, self.config = model, config
        self.device = torch.device(device)
        self.cache = cache
        self.max_state_length = config.data.max_state_length if max_state_length is None else int(max_state_length)
        self.max_option_length = config.data.max_option_length if max_option_length is None else int(max_option_length)
        if min(self.max_state_length, self.max_option_length) < 1:
            raise ValueError("maximum token lengths must be positive")
        self.truncation = truncation
        self.option_chunk_size, self.option_batch_tokens = option_chunk_size, option_batch_tokens
        self.state_chunk_size, self.state_batch_tokens = state_chunk_size, state_batch_tokens
        self._options = BoundedCache(option_cache_size, option_cache_max_bytes)
        self._sets = BoundedCache(set_cache_size, set_cache_max_bytes)
        self._states = BoundedCache(state_cache_size, state_cache_max_bytes)
        self._lock = RLock()
        self.checkpoint_dir: Path | None = None
        self.quantize = quantize
        self.compile_core = bool(compile_core)
        self._compiled_decide = None
        if quantize is None:
            self.model.to(self.device)
        self.model.eval().requires_grad_(False)
        if self.compile_core:
            try:
                self._compiled_decide = torch.compile(
                    self.model.decide_from_option_cache, dynamic=False
                )
            except Exception:
                self._compiled_decide = None

    @classmethod
    def from_pretrained(cls, checkpoint="LIQIIIII/Jeva-4B", *, device="cuda",
                        merge_lora=True, cache="hybrid", dtype=None, revision=None,
                        local_files_only=False, trust_remote_code=False,
                        quantize=None, **kwargs):

        from ._runtime.model import System1DecisionModel

        if not isinstance(merge_lora, bool):
            raise ValueError("merge_lora must be a bool")
        if quantize not in {None, "4bit", "8bit"}:
            raise ValueError("quantize must be None, '4bit', or '8bit'")
        target_device = torch.device(device)
        if target_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; select device='cpu' or use a CUDA environment")
        if quantize is not None and target_device.type != "cuda":
            raise ValueError("quantize requires device='cuda'")
        directory = Path(checkpoint).expanduser()
        if not directory.is_dir():
            if directory.is_absolute() or str(checkpoint).startswith("."):
                raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
            from huggingface_hub import snapshot_download
            from .download import INFERENCE_FILES
            directory = Path(snapshot_download(
                repo_id=str(checkpoint), revision=revision, local_files_only=local_files_only,
                allow_patterns=INFERENCE_FILES,
            ))
        directory = directory.resolve()
        config = load_config(directory / "config.yaml")
        cfg = config.model
        for field in ("backbone", "tokenizer_path", "head_path", "adapter_path"):
            value = getattr(cfg, field)
            if value and not Path(value).is_absolute() and (directory / value).exists():
                setattr(cfg, field, str((directory / value).resolve()))
        if (directory / "backbone_full").is_dir():
            cfg.backbone = str(directory / "backbone_full")
            cfg.tune_mode, cfg.adapter_path = "full", None
        elif (directory / "backbone_adapter").is_dir():
            cfg.adapter_path = str(directory / "backbone_adapter")
            cfg.tune_mode = "lora"
        if (directory / "tokenizer").is_dir():
            cfg.tokenizer_path = str(directory / "tokenizer")
        cfg.head_path = str(directory / "head.pt")
        if not Path(cfg.head_path).is_file():
            raise FileNotFoundError(f"Missing checkpoint head: {cfg.head_path}")
        if cfg.tune_mode == "lora" and not cfg.adapter_path:
            raise FileNotFoundError("LoRA checkpoint is missing backbone_adapter")
        cfg.gradient_checkpointing = False
        cfg.trust_remote_code = bool(trust_remote_code)
        cfg.local_files_only = bool(local_files_only)
        if dtype is not None:
            dtype_name = str(dtype).removeprefix("torch.")
            if dtype_name not in {"float32", "float16", "bfloat16"}:
                raise ValueError("dtype must be float32, float16, or bfloat16")
            cfg.torch_dtype = dtype_name
        model = System1DecisionModel(cfg, quantize=quantize, device=target_device).eval()
        if merge_lora and cfg.tune_mode == "lora" and quantize is None:
            with torch.no_grad():
                model.backbone.model = model.backbone.model.merge_and_unload(safe_merge=True)
        instance = cls(model, config, device=target_device, cache=cache, quantize=quantize, **kwargs)
        instance.checkpoint_dir = directory
        instance.lora_merged = bool(merge_lora and cfg.tune_mode == "lora" and quantize is None)
        return instance

    def _sync(self):
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def clear_cache(self):
        with self._lock:
            self._options.clear()
            self._sets.clear()
            self._states.clear()

    def cache_info(self):
        with self._lock:
            return {
                "mode": self.cache,
                "options": self._options.info(),
                "sets": self._sets.info(),
                "states": self._states.info(),
            }

    @staticmethod
    def _state_text(state):
        if isinstance(state, str):
            return state
        return json.dumps(state, ensure_ascii=False, sort_keys=True, allow_nan=False)

    def _batch(self, rows):
        batch = self.model.tokenizer.pad(rows, padding=True, return_tensors="pt")
        return {key: value.to(self.device) for key, value in batch.items()}

    def _chunk_rows(self, rows, chunk_size, token_budget):
        outputs = []
        start = 0
        while start < len(rows):
            end, longest = start, 0
            while end < len(rows) and end - start < chunk_size:
                candidate = max(longest, len(rows[end]["input_ids"]))
                if end > start and candidate * (end + 1 - start) > token_budget:
                    break
                longest = candidate
                end += 1
            outputs.append(rows[start:end])
            start = end
        return outputs

    def _encode_state_rows(self, rows):
        encoded = [
            self.model.encode_state_inputs(self._batch(chunk))
            for chunk in self._chunk_rows(rows, self.state_chunk_size, self.state_batch_tokens)
        ]
        return torch.cat(encoded, dim=0)

    def _encode_options(self, rows):
        encoded = [
            self.model.encode_option_inputs(self._batch(chunk))
            for chunk in self._chunk_rows(rows, self.option_chunk_size, self.option_batch_tokens)
        ]
        return torch.cat(encoded, dim=0)

    @staticmethod
    def _criteria(criteria):
        if isinstance(criteria, Mapping):
            keys = [str(key) for key in criteria]
            descriptions = list(criteria.values())
            if not all(isinstance(value, str) and value.strip() for value in descriptions):
                raise ValueError("criteria descriptions must be nonempty strings")
            texts = [f"{key}: {description}" for key, description in zip(keys, descriptions, strict=True)]
        elif isinstance(criteria, Sequence) and not isinstance(criteria, (str, bytes)):
            keys = list(criteria)
            texts = keys.copy()
        else:
            raise TypeError("criteria must be a mapping or a sequence of strings")
        if not keys or not all(isinstance(key, str) and key.strip() for key in keys):
            raise ValueError("criteria must contain nonempty string labels")
        if len(set(keys)) != len(keys):
            raise ValueError("criteria labels must be unique")
        return keys, texts

    def _question(self, question):
        if isinstance(question, Noul):
            return question.instructions, "noul", ["false", "true"], ["false", "true"], False, None, 1
        if not isinstance(question, (Choice, Score)):
            raise TypeError("questions must be Choice, Noul, or Score objects")
        labels, options = self._criteria(question.criteria)
        if isinstance(question, Choice):
            return question.instructions, "choice", labels, options, question.allow_null, None, None
        values = question.values
        if values is None:
            if isinstance(question.criteria, Mapping):
                try:
                    values = [float(key) for key in labels]
                except ValueError:
                    values = list(range(len(labels)))
            else:
                values = list(range(len(labels)))
        return question.instructions, "score", labels, options, False, values, None

    def _prepare(self, state, instructions, kind, labels, options, allow_null, values, positive_index):
        if not isinstance(instructions, str) or not instructions.strip():
            raise ValueError("instructions must be a nonempty string")
        if kind not in {"choice", "noul", "score"}:
            raise ValueError("decision_type must be choice, noul, or score")
        if not isinstance(allow_null, bool):
            raise ValueError("allow_null must be a bool")
        if not options or not all(isinstance(text, str) and text.strip() for text in options):
            raise ValueError("options must contain nonempty strings")
        if kind == "noul":
            if len(options) != 2 or type(positive_index) is not int or positive_index not in {0, 1}:
                raise ValueError("noul requires two options and positive_index 0 or 1")
        if kind == "score":
            if values is not None and (not isinstance(values, Sequence) or isinstance(values, (str, bytes))):
                raise ValueError("score values must be a sequence of finite numbers")
            values = list(range(len(options))) if values is None else list(values)
            if len(values) != len(options) or not all(
                isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                for v in values
            ):
                raise ValueError("score values must match criteria and be finite")
        if self.model.training:
            raise RuntimeError("Inference requires eval mode; clear caches after training before reuse")
        state_rows, state_lengths, truncated_state = tokenize_texts(
            self.model.tokenizer, [state_question_text(state, instructions, kind)],
            self.max_state_length, self.truncation, field="state",
        )
        texts = [option_text(text, kind) for text in options]
        rows, lengths, truncated_options = tokenize_texts(
            self.model.tokenizer, texts, self.max_option_length, self.truncation, field="option",
        )
        keys = [tuple(row["input_ids"]) for row in rows]
        return _PreparedDecision(
            kind=kind, labels=labels, options=list(options), allow_null=allow_null,
            values=values, positive_index=positive_index,
            state_row=state_rows[0], state_length=state_lengths[0], truncated_state=truncated_state,
            option_rows=rows, option_lengths=lengths, truncated_options=truncated_options,
            keys=keys, set_key=tuple(keys),
        )

    def _decide(self, c, option_cache):
        compiled = self._compiled_decide
        if compiled is None:
            return self.model.decide_from_option_cache(c, option_cache)
        try:
            return compiled(c, option_cache)
        except Exception:
            self._compiled_decide = None
            return self.model.decide_from_option_cache(c, option_cache)

    def _decide_batch(self, prepared: list[_PreparedDecision]) -> list[Answer]:
        if not prepared:
            return []
        use_cache = self.cache != "none"
        self._sync()
        started = perf_counter()

        state_keys = [tuple(row["input_ids"]) for row in (p.state_row for p in prepared)]
        state_vectors = {}
        if use_cache:
            for key in dict.fromkeys(state_keys):
                memory = self._states.get(key)
                if memory is not None:
                    state_vectors[key] = memory
        pending_states = []
        seen = set(state_vectors)
        for prepared_item, key in zip(prepared, state_keys, strict=True):
            if key in seen:
                continue
            seen.add(key)
            pending_states.append((prepared_item, key))
        if pending_states:
            encoded = self._encode_state_rows([item.state_row for item, _ in pending_states])
            for (_, key), vector in zip(pending_states, encoded, strict=True):
                vector = vector.unsqueeze(0)
                state_vectors[key] = vector
                if use_cache:
                    self._states.put(key, vector)

        option_memory = {}
        encoded_option_tokens = {}
        if self.cache == "hybrid":
            keys = list(dict.fromkeys(key for item in prepared for key in item.keys))
            resolved = {}
            missing_keys = set()
            for key in keys:
                memory = self._options.get(key)
                if memory is None:
                    missing_keys.add(key)
                else:
                    resolved[key] = memory
            if missing_keys:
                row_by_key = {}
                for item in prepared:
                    for key, row in zip(item.keys, item.option_rows, strict=True):
                        if key in missing_keys:
                            row_by_key.setdefault(key, row)
                order = [key for key in keys if key in missing_keys]
                encoded = self._encode_options([row_by_key[key] for key in order])
                memories = self.model.compile_option_memory(encoded)
                for key, memory in zip(order, memories, strict=True):
                    resolved[key] = memory
                    self._options.put(key, memory)
            for item in prepared:
                set_cache = self._sets.get(item.set_key)
                if set_cache is None:
                    set_cache = self.model.assemble_option_memory([resolved[key] for key in item.keys])
                    self._sets.put(item.set_key, set_cache)
                    encoded_option_tokens[id(item)] = sum(
                        length for key, length in zip(item.keys, item.option_lengths, strict=True)
                        if key in missing_keys
                    )
                else:
                    encoded_option_tokens[id(item)] = 0
                option_memory[id(item)] = set_cache
        else:
            for item in prepared:
                set_cache = self._sets.get(item.set_key) if self.cache == "set" else None
                if set_cache is not None:
                    option_memory[id(item)] = set_cache
                    encoded_option_tokens[id(item)] = 0
                    continue
                encoded = self._encode_options(item.option_rows)
                mask = torch.ones((1, len(item.options)), dtype=torch.bool, device=self.device)
                set_cache = self.model.prepare_option_kv_cache(encoded.unsqueeze(0), mask)
                if self.cache == "set":
                    self._sets.put(item.set_key, set_cache)
                option_memory[id(item)] = set_cache
                encoded_option_tokens[id(item)] = sum(item.option_lengths)

        answers = []
        for item, key in zip(prepared, state_keys, strict=True):
            out = self._decide(state_vectors[key], option_memory[id(item)])
            q = out["q"][0].float().cpu().tolist()
            p_null = float(out["p_null"][0].float().cpu()) if item.allow_null else 0.0
            if (len(q) != len(item.options)
                    or not all(math.isfinite(value) and 0 <= value <= 1 for value in [*q, p_null])
                    or not math.isclose(sum(q), 1.0, abs_tol=1e-5)):
                raise RuntimeError("Model returned invalid probabilities")
            probabilities = [(1.0 - p_null) * value for value in q]
            best = max(range(len(q)), key=q.__getitem__)
            encoded = encoded_option_tokens[id(item)]
            usage = Usage(
                input_tokens=item.state_length + encoded,
                requested_input_tokens=item.state_length + sum(item.option_lengths),
                encoded_state_tokens=item.state_length,
                encoded_option_tokens=encoded,
                cached_option_tokens=sum(item.option_lengths) - encoded,
                truncated_tokens=item.truncated_state + item.truncated_options,
            )
            answers.append(Answer(
                kind=item.kind, probabilities=dict(zip(item.labels, probabilities, strict=True)),
                conditional_probabilities=dict(zip(item.labels, q, strict=True)),
                null_probability=p_null,
                choice=(None if item.allow_null and p_null >= probabilities[best] else item.labels[best])
                if item.kind == "choice" else None,
                noul=q[item.positive_index] if item.kind == "noul" else None,
                score=sum(p * float(v) for p, v in zip(q, item.values, strict=True))
                if item.kind == "score" else None,
                confidence=max(p_null, max(probabilities)),
                candidate_probabilities=tuple(q), usage=usage,
                latency_ms=(perf_counter() - started) * 1000,
            ))
        self._sync()
        return answers

    def _predict(self, state, instructions, kind, labels, options, allow_null, values, positive_index):
        prepared = self._prepare(state, instructions, kind, labels, options, allow_null, values, positive_index)
        return self._decide_batch([prepared])[0]

    def prime_options(self, options, *, kind="choice"):
        if kind not in {"choice", "noul", "score"}:
            raise ValueError("decision_type must be choice, noul, or score")
        if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
            raise TypeError("options must be a sequence of strings")
        options = list(options)
        if not options or not all(isinstance(text, str) and text.strip() for text in options):
            raise ValueError("options must contain nonempty strings")
        if self.cache == "none":
            raise ValueError("priming requires cache='set' or cache='hybrid'")
        with self._lock:
            if self.model.training:
                raise RuntimeError("Inference requires eval mode")
            self._sync()
            started = perf_counter()
            rows, lengths, truncated = tokenize_texts(
                self.model.tokenizer, [option_text(text, kind) for text in options],
                self.max_option_length, self.truncation, field="option",
            )
            keys = [tuple(row["input_ids"]) for row in rows]
            unique = list(dict.fromkeys(keys))
            set_key = tuple(keys)
            total_tokens = sum(lengths)
            if self.cache == "set":
                encoded = self._encode_options(rows)
                mask = torch.ones((1, len(rows)), dtype=torch.bool, device=self.device)
                set_cache = self.model.prepare_option_kv_cache(encoded.unsqueeze(0), mask)
                self._sets.put(set_key, set_cache)
                result = {
                    "kind": kind, "options": len(unique), "encoded_options": len(unique),
                    "encoded_tokens": total_tokens, "cached_tokens": 0,
                    "truncated_tokens": truncated, "set_cached": True,
                }
            else:
                length_by_key = {}
                row_by_key = {}
                for key, row, length in zip(keys, rows, lengths, strict=True):
                    row_by_key.setdefault(key, row)
                    length_by_key.setdefault(key, length)
                missing = [key for key in unique if self._options.get(key) is None]
                encoded_tokens = sum(length_by_key[key] for key in missing)
                if missing:
                    encoded = self._encode_options([row_by_key[key] for key in missing])
                    memories = self.model.compile_option_memory(encoded)
                    for key, memory in zip(missing, memories, strict=True):
                        self._options.put(key, memory)
                set_cached = False
                if self._sets.get(set_key) is None:
                    memories = [self._options.get(key) for key in keys]
                    if all(memory is not None for memory in memories):
                        set_cache = self.model.assemble_option_memory(memories)
                        self._sets.put(set_key, set_cache)
                        set_cached = True
                result = {
                    "kind": kind, "options": len(unique), "encoded_options": len(missing),
                    "encoded_tokens": encoded_tokens,
                    "cached_tokens": total_tokens - encoded_tokens,
                    "truncated_tokens": truncated, "set_cached": set_cached,
                }
            self._sync()
            result["latency_ms"] = (perf_counter() - started) * 1000
            return result

    def prime_question(self, question):
        _, kind, _, options, _, _, _ = self._question(question)
        return self.prime_options(options, kind=kind)

    def prime(self, questions):
        with self._lock:
            named = isinstance(questions, Mapping)
            items = list(questions.items()) if named else list(enumerate(questions))
            if not items:
                raise ValueError("questions must not be empty")
            results = {}
            for name, question in items:
                results[name] = self.prime_question(question)
            return results

    def predict(self, state, question, decision_type, options, *, allow_null=False,
                values=None, positive_index=1):

        with self._lock:
            if not isinstance(options, Sequence) or isinstance(options, (str, bytes)):
                raise TypeError("options must be a sequence of strings")
            options = list(options)
            return self._predict(self._state_text(state), question, decision_type,
                                 [str(i) for i in range(len(options))], options,
                                 allow_null, values, positive_index)

    def _prepare_request(self, state, questions):
        named = isinstance(questions, Mapping)
        items = list(questions.items()) if named else list(enumerate(questions))
        if not items:
            raise ValueError("questions must not be empty")
        if named and not all(isinstance(name, str) for name, _ in items):
            raise TypeError("question names must be strings")
        text = self._state_text(state)
        prepared = [(name, self._prepare(text, *self._question(question))) for name, question in items]
        return named, prepared

    @staticmethod
    def _aggregate(answers):
        usage = Usage()
        for answer in answers:
            for key in usage.__dataclass_fields__:
                setattr(usage, key, getattr(usage, key) + getattr(answer.usage, key))
        return usage

    def system_one(self, state, questions):

        with self._lock:
            named, prepared = self._prepare_request(state, questions)
            started = perf_counter()
            answers = self._decide_batch([item for _, item in prepared])
            usage = self._aggregate(answers)
            latency = (perf_counter() - started) * 1000
            if named:
                return DecisionResponse(dict(zip([name for name, _ in prepared], answers, strict=True)),
                                        usage, latency)
            return DecisionResponse(answers, usage, latency)

    def system_one_many(self, requests):

        with self._lock:
            batches = []
            flat = []
            for state, questions in requests:
                named, prepared = self._prepare_request(state, questions)
                batches.append((named, prepared))
                flat.extend(item for _, item in prepared)
            started = perf_counter()
            answers = self._decide_batch(flat)
            responses = []
            offset = 0
            for named, prepared in batches:
                count = len(prepared)
                chunk = answers[offset:offset + count]
                offset += count
                usage = self._aggregate(chunk)
                latency = (perf_counter() - started) * 1000
                if named:
                    response = DecisionResponse(
                        dict(zip([name for name, _ in prepared], chunk, strict=True)), usage, latency
                    )
                else:
                    response = DecisionResponse(chunk, usage, latency)
                responses.append(response)
            return responses

    def rank(self, state, candidates, *, instructions="Rank candidates by relevance to the state.",
             allow_null=False, top_k=None):

        if top_k is not None and (isinstance(top_k, bool) or not isinstance(top_k, int) or top_k < 1):
            raise ValueError("top_k must be a positive integer")
        response = self.system_one(state, [Choice(instructions, candidates, allow_null)])
        answer = response.answers[0]
        ranking = [RankedItem(label, probability, i)
                   for i, (label, probability) in enumerate(answer.probabilities.items())]
        ranking.sort(key=lambda item: -item.probability)
        return RankResult(ranking[:top_k], answer.null_probability, response.usage, response.latency_ms)
