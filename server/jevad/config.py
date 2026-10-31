from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value in (None, "") else int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value in (None, "") else float(value)


def _env_optional_int(name: str) -> int | None:
    value = os.environ.get(name)
    return None if value in (None, "") else int(value)


def _local_checkpoint(hub_id: str) -> str:
    relative = Path("models") / hub_id.rsplit("/", 1)[-1]
    for base in (Path.cwd(), *Path.cwd().parents):
        candidate = base / relative
        if (candidate / "config.yaml").is_file():
            return str(candidate)
    return hub_id


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value in (None, ""):
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    model: str = "LIQIIIII/Jeva-4B"
    model_name: str = "LiQiiiii/Jeva-4B"
    device: str = "cuda"
    cache: str = "hybrid"
    api_key: str | None = None
    queue_size: int = 32
    request_timeout: float = 180.0
    option_chunk_size: int = 16
    option_batch_tokens: int = 16384
    max_state_length: int | None = None
    max_option_length: int | None = None
    local_files_only: bool = False
    quantize: str | None = None
    batch_window_ms: float = 5.0
    max_batch: int = 8
    compile_core: bool = False
    host: str = "0.0.0.0"
    port: int = 8000

    @classmethod
    def from_env(cls) -> Settings:
        model = os.environ.get("JEVA_MODEL") or _local_checkpoint(cls.model)
        default_name = Path(model).name if os.sep in model or model.startswith(".") else model
        return cls(
            model=model,
            model_name=os.environ.get("JEVA_MODEL_NAME", default_name),
            device=os.environ.get("JEVA_DEVICE", cls.device),
            cache=os.environ.get("JEVA_CACHE", cls.cache),
            api_key=os.environ.get("JEVA_API_KEY") or None,
            queue_size=_env_int("JEVA_QUEUE_SIZE", cls.queue_size),
            request_timeout=_env_float("JEVA_REQUEST_TIMEOUT", cls.request_timeout),
            option_chunk_size=_env_int("JEVA_OPTION_CHUNK_SIZE", cls.option_chunk_size),
            option_batch_tokens=_env_int("JEVA_OPTION_BATCH_TOKENS", cls.option_batch_tokens),
            max_state_length=_env_optional_int("JEVA_MAX_STATE_LENGTH"),
            max_option_length=_env_optional_int("JEVA_MAX_OPTION_LENGTH"),
            local_files_only=_env_bool("JEVA_LOCAL_FILES_ONLY", cls.local_files_only),
            quantize=os.environ.get("JEVA_QUANTIZE") or None,
            batch_window_ms=_env_float("JEVA_BATCH_WINDOW_MS", cls.batch_window_ms),
            max_batch=_env_int("JEVA_MAX_BATCH", cls.max_batch),
            compile_core=_env_bool("JEVA_COMPILE", cls.compile_core),
            host=os.environ.get("JEVA_HOST", cls.host),
            port=_env_int("JEVA_PORT", cls.port),
        )
