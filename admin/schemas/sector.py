from datetime import datetime
from pydantic import BaseModel


class SectorResponse(BaseModel):
    id: int
    name: str
    display_name: str
    description: str | None
    icon: str | None
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class SectorListResponse(BaseModel):
    sectors: list[SectorResponse]
    total: int
