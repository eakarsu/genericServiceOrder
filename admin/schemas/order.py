from datetime import datetime
from pydantic import BaseModel, Field


class OrderUpdate(BaseModel):
    customer_name: str | None = None
    customer_phone: str | None = None
    customer_email: str | None = None
    sector: str | None = None
    status: str | None = None
    items: list[dict] | None = None
    total_amount: float | None = None
    notes: str | None = None


class OrderResponse(BaseModel):
    id: int
    customer_name: str
    customer_phone: str | None
    customer_email: str | None
    sector: str
    status: str
    items: list[dict] | None
    total_amount: float
    notes: str | None
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class OrderListResponse(BaseModel):
    orders: list[OrderResponse]
    total: int
    page: int
    page_size: int
    total_pages: int


class BulkDeleteRequest(BaseModel):
    ids: list[int] = Field(min_length=1)


class BulkUpdateRequest(BaseModel):
    ids: list[int] = Field(min_length=1)
    status: str


class BulkResponse(BaseModel):
    message: str
    affected: int
