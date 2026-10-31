from __future__ import annotations

import json
from typing import Any


def chunk_text(text: str, size: int = 256) -> list[str]:
    return [text[index : index + size] for index in range(0, len(text), size)]


def sse(payload: Any) -> str:
    if hasattr(payload, "model_dump_json"):
        return f"data: {payload.model_dump_json()}\n\n"
    return f"data: {json.dumps(payload)}\n\n"


def sse_headers() -> dict[str, str]:
    return {
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no",
    }
