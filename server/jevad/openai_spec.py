from __future__ import annotations

import json
from typing import Any

from fastapi import HTTPException
from pydantic import ValidationError

from . import schemas as S


def strip_fences(text: str) -> str:
    raw = text.strip()
    if not raw.startswith("```"):
        return raw
    lines = raw.splitlines()[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_decision_spec(text: str) -> S.DecideRequest:
    try:
        data = json.loads(strip_fences(text))
    except json.JSONDecodeError as exc:
        raise HTTPException(
            400,
            "Request content must be a JSON object with 'state' and 'questions': "
            f"{exc.msg} at line {exc.lineno} column {exc.colno}",
        ) from exc
    if not isinstance(data, dict):
        raise HTTPException(
            400, "Request content must be a JSON object, not a JSON array or scalar."
        )
    try:
        return S.DecideRequest.model_validate(data)
    except ValidationError as exc:
        raise HTTPException(400, f"Invalid decision specification: {exc}") from exc


def message_text(message: S.ChatMessage) -> str:
    if isinstance(message.content, str):
        return message.content
    if isinstance(message.content, list):
        parts = [
            part.get("text", "")
            for part in message.content
            if isinstance(part, dict) and part.get("type") in (None, "text")
        ]
        return "".join(parts)
    return ""


def openai_usage(usage: Any) -> S.OpenAIUsage:
    prompt = int(getattr(usage, "input_tokens", 0) or 0)
    completion = int(getattr(usage, "output_tokens", 0) or 0)
    return S.OpenAIUsage(
        prompt_tokens=prompt,
        completion_tokens=completion,
        total_tokens=prompt + completion,
    )
