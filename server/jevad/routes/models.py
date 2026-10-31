from __future__ import annotations

from fastapi import APIRouter, HTTPException

from .. import schemas as S
from ..deps import CreatedDep, SettingsDep

router = APIRouter(prefix="/v1", tags=["models"])


@router.get("/models", response_model=S.ModelList)
async def list_models(settings: SettingsDep, created: CreatedDep):
    return S.ModelList(data=[S.ModelCard(id=settings.model_name, created=created)])


@router.get("/models/{model_id}", response_model=S.ModelCard)
async def get_model(model_id: str, settings: SettingsDep, created: CreatedDep):
    if model_id != settings.model_name:
        raise HTTPException(404, f"Model '{model_id}' not found.")
    return S.ModelCard(id=settings.model_name, created=created)
