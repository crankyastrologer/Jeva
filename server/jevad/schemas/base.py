from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class OpenModel(BaseModel):
    model_config = ConfigDict(extra="allow")
