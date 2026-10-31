from __future__ import annotations

from typing import Annotated, Any

from fastapi import Depends, HTTPException, Request

from .config import Settings
from .runtime import InferenceEngine


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_created(request: Request) -> int:
    return request.app.state.created


def get_engine(request: Request) -> InferenceEngine:
    engine = request.app.state.engine
    if engine is None:
        raise HTTPException(503, "Model is not loaded yet.")
    return engine


def get_model(request: Request) -> Any:
    return request.app.state.model


SettingsDep = Annotated[Settings, Depends(get_settings)]
CreatedDep = Annotated[int, Depends(get_created)]
EngineDep = Annotated[InferenceEngine, Depends(get_engine)]
ModelDep = Annotated[Any, Depends(get_model)]
