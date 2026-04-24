from sqlalchemy import Column, Integer, String, Float, DateTime, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from admin.database import Base


class Order(Base):
    __tablename__ = "orders"

    id = Column(Integer, primary_key=True, index=True)
    customer_name = Column(String(255), nullable=False)
    customer_phone = Column(String(50), nullable=True)
    customer_email = Column(String(255), nullable=True)
    sector = Column(String(100), nullable=False, index=True)
    status = Column(String(50), default="pending", nullable=False, index=True)
    items = Column(JSONB, default=[])
    total_amount = Column(Float, default=0.0)
    notes = Column(Text, nullable=True)
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), server_default=func.now(), onupdate=func.now())
