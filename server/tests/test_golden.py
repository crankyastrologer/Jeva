from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from jevad.app import create_app
from jevad.config import Settings
from jevad.stub import StubJeva

from tests.golden_cases import collect

GOLDEN = Path(__file__).with_name("golden") / "responses.json"
UNSTABLE = {"id", "created", "latency_ms", "system_fingerprint"}


def normalize(value):
    if isinstance(value, dict):
        return {key: normalize(item) for key, item in value.items() if key not in UNSTABLE}
    if isinstance(value, list):
        return [normalize(item) for item in value]
    return value


@pytest.fixture(scope="module")
def snapshot() -> dict:
    app = create_app(
        settings=Settings(model="stub", model_name="jevad-stub", device="cpu", queue_size=4),
        model=StubJeva(),
    )
    with TestClient(app) as client:
        return normalize(collect(client))


def test_golden_snapshot(snapshot):
    expected = normalize(json.loads(GOLDEN.read_text()))
    assert snapshot == expected


def test_golden_covers_all_cases(snapshot):
    assert set(snapshot) == {
        "healthz",
        "models",
        "model",
        "decide",
        "rank",
        "cache",
        "prime",
        "chat",
        "completions",
        "chat_stream",
        "error_validation",
        "error_not_found",
    }
