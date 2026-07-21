from datetime import datetime

from pydantic import BaseModel, Field


class ResourceCreate(BaseModel):
    name: str = Field(min_length=1, max_length=255)
    skills: list[str] = Field(min_length=1)
    service_area_codes: list[str] = Field(min_length=1)
    travel_buffer_minutes: int = Field(default=30, ge=0, le=240)


class InventoryUpsert(BaseModel):
    sku: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=255)
    on_hand: int = Field(ge=0)
    unit_price_cents: int = Field(ge=0)


class QuoteItem(BaseModel):
    sku: str = Field(min_length=1, max_length=100)
    quantity: int = Field(gt=0)


class QuoteCreate(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=200)
    customer_name: str = Field(min_length=1, max_length=255)
    customer_email: str | None = Field(default=None, max_length=255)
    customer_phone: str | None = Field(default=None, max_length=50)
    service_code: str = Field(min_length=1, max_length=100)
    required_skills: list[str] = Field(min_length=1)
    service_area_code: str = Field(min_length=1, max_length=100)
    items: list[QuoteItem] = Field(default_factory=list)
    tax_basis_points: int = Field(default=0, ge=0, le=10_000)
    expires_at: datetime


class BookingRequest(BaseModel):
    start: datetime
    end: datetime
    resource_id: str | None = None


class ReassignRequest(BaseModel):
    resource_id: str


class JobStatusRequest(BaseModel):
    status: str


class ChangeCreate(BaseModel):
    idempotency_key: str = Field(min_length=8, max_length=200)
    description: str = Field(min_length=1, max_length=4000)
    amount_delta_cents: int


class ChangeDecision(BaseModel):
    approve: bool
    reason: str = Field(min_length=1, max_length=2000)


class CancellationRequest(BaseModel):
    reason: str = Field(min_length=1, max_length=2000)


class OfflineReplayRequest(BaseModel):
    device_id: str = Field(min_length=1, max_length=200)
    operation_id: str = Field(min_length=1, max_length=200)
    order_id: str
    expected_version: int = Field(ge=1)
    operation: str
    payload: dict = Field(default_factory=dict)
