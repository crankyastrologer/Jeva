from __future__ import annotations

import zlib

import torch
from jeva import Choice, Jeva, Noul
from jeva._runtime.config import Config
from torch import nn


class FakeTokenizer:
    def __call__(self, texts, padding=False, truncation=False, max_length=None):
        ids = [[len(text) + 1, 1 + (zlib.crc32(text.encode()) % 97)] for text in texts]
        return {"input_ids": ids, "attention_mask": [[1] * len(row) for row in ids]}

    def pad(self, rows, padding=True, return_tensors="pt"):
        width = max(len(row["input_ids"]) for row in rows)
        ids = [row["input_ids"] + [0] * (width - len(row["input_ids"])) for row in rows]
        mask = [row["attention_mask"] + [0] * (width - len(row["attention_mask"])) for row in rows]
        return {"input_ids": torch.tensor(ids), "attention_mask": torch.tensor(mask)}


class Blob:
    nbytes = 8

    def __init__(self, count=1):
        self.count = count


class FakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = FakeTokenizer()
        self.state_calls = 0
        self.option_calls = 0
        self.scale = nn.Parameter(torch.ones(1))

    def encode_state_inputs(self, batch):
        self.state_calls += batch["input_ids"].shape[0]
        return torch.ones((batch["input_ids"].shape[0], 4))

    def encode_option_inputs(self, batch):
        self.option_calls += batch["input_ids"].shape[0]
        return torch.ones((batch["input_ids"].shape[0], 4))

    def compile_option_memory(self, u, *, detach=True):
        return [Blob() for _ in range(u.shape[0])]

    def assemble_option_memory(self, memories):
        return Blob(len(memories))

    def prepare_option_kv_cache(self, u, mask, *, detach=True):
        return Blob(int(u.shape[1]))

    def decide_from_option_cache(self, c, cache):
        q = torch.full((1, cache.count), 1.0 / cache.count)
        return {"q": q, "p_null": torch.zeros(1)}


def make(**kwargs):
    model = FakeModel()
    return Jeva(model, Config(), device="cpu", **kwargs), model


def test_state_and_option_cache_reuse():
    jeva, model = make()
    jeva.system_one("ticket text", {"q": Noul("Is it about billing?")})
    jeva.system_one("ticket text", {"q": Noul("Is it about billing?")})
    assert model.state_calls == 1
    assert model.option_calls == 2
    assert jeva.cache_info()["states"]["hits"] >= 1


def test_batch_deduplicates_options_across_requests():
    jeva, model = make()
    criteria = {"x": "X.", "y": "Y."}
    responses = jeva.system_one_many(
        [
            ("first state", {"q": Choice("Pick one.", criteria)}),
            ("second state", {"q": Choice("Pick one.", criteria)}),
        ]
    )
    assert len(responses) == 2
    assert model.state_calls == 2
    assert model.option_calls == 2


def test_batch_keeps_state_per_question():
    jeva, model = make()
    jeva.system_one("shared state", {"a": Noul("Question A?"), "b": Noul("Question B?")})
    assert model.state_calls == 2


def test_cache_none_disables_reuse():
    jeva, model = make(cache="none")
    jeva.system_one("state", {"q": Noul("Q?")})
    jeva.system_one("state", {"q": Noul("Q?")})
    assert model.state_calls == 2
    assert model.option_calls == 4


def test_prime_then_decide_uses_cached_options():
    jeva, model = make()
    criteria = {"x": "X.", "y": "Y."}
    primed = jeva.prime({"q": Choice("Pick one.", criteria)})
    assert primed["q"]["set_cached"] is True
    assert primed["q"]["encoded_options"] == 2
    assert model.option_calls == 2
    response = jeva.system_one("new state", {"q": Choice("Pick one.", criteria)})
    assert response.answers["q"].usage.encoded_option_tokens == 0
    assert model.option_calls == 2
    assert model.state_calls == 1


def test_primed_options_reused_across_instructions():
    jeva, model = make()
    jeva.prime_options(["alpha", "beta"], kind="choice")
    assert model.option_calls == 2
    response = jeva.system_one("state", {"q": Choice("Different instructions", ["alpha", "beta"])})
    assert response.answers["q"].usage.encoded_option_tokens == 0
    assert model.option_calls == 2


def test_prime_requires_cache():
    jeva, _ = make(cache="none")
    try:
        jeva.prime_options(["alpha"])
    except ValueError as exc:
        assert "priming requires" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_list_questions_and_many_requests():
    jeva, model = make()
    responses = jeva.system_one_many(
        [
            ("state one", [Noul("Q1?"), Choice("Q2?", ["a", "b", "c"])]),
            ("state two", {"q": Noul("Q3?")}),
        ]
    )
    assert isinstance(responses[0].answers, list)
    assert isinstance(responses[1].answers, dict)
    assert model.state_calls == 3
    assert responses[0].usage.input_tokens > 0
