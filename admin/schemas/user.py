from datetime import datetime
from pydantic import BaseModel, EmailStr, Field


class UserBase(BaseModel):
    email: EmailStr
    name: str = Field(min_length=1, max_length=255)
    phone: str | None = None


class UserCreate(UserBase):
    password: str = Field(min_length=8, max_length=128)
    role_id: int = 3  # default viewer


class UserUpdate(BaseModel):
    name: str | None = None
    phone: str | None = None
    role_id: int | None = None
    is_active: bool | None = None


class ProfileUpdate(BaseModel):
    name: str | None = None
    phone: str | None = None


class UserResponse(BaseModel):
    id: int
    email: str
    name: str
    phone: str | None
    role_id: int
    role_name: str | None = None
    is_active: bool
    is_email_verified: bool
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    users: list[UserResponse]
    total: int
    page: int
    page_size: int
    total_pages: int
