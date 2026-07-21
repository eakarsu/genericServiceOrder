from __future__ import annotations

import uuid

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import relationship

from admin.database import Base

JSON_TYPE = JSON().with_variant(JSONB, "postgresql")


def new_id() -> str:
    return str(uuid.uuid4())


class ServiceResource(Base):
    __tablename__ = "service_resources"

    id = Column(String(36), primary_key=True, default=new_id)
    name = Column(String(255), nullable=False)
    skills = Column(JSON_TYPE, nullable=False, default=list)
    service_area_codes = Column(JSON_TYPE, nullable=False, default=list)
    travel_buffer_minutes = Column(Integer, nullable=False, default=30)
    is_active = Column(Boolean, nullable=False, default=True)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())


class InventoryItem(Base):
    __tablename__ = "service_inventory"

    id = Column(String(36), primary_key=True, default=new_id)
    sku = Column(String(100), nullable=False, unique=True, index=True)
    name = Column(String(255), nullable=False)
    on_hand = Column(Integer, nullable=False, default=0)
    reserved = Column(Integer, nullable=False, default=0)
    unit_price_cents = Column(Integer, nullable=False, default=0)
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    __table_args__ = (
        CheckConstraint("on_hand >= 0", name="inventory_on_hand_nonnegative"),
        CheckConstraint("reserved >= 0 AND reserved <= on_hand", name="inventory_reserved_valid"),
        CheckConstraint("unit_price_cents >= 0", name="inventory_price_nonnegative"),
    )


class ServiceOrder(Base):
    __tablename__ = "service_orders"

    id = Column(String(36), primary_key=True, default=new_id)
    idempotency_key = Column(String(200), nullable=False, unique=True)
    customer_name = Column(String(255), nullable=False)
    customer_email = Column(String(255), nullable=True)
    customer_phone = Column(String(50), nullable=True)
    service_code = Column(String(100), nullable=False, index=True)
    required_skills = Column(JSON_TYPE, nullable=False, default=list)
    service_area_code = Column(String(100), nullable=False, index=True)
    currency = Column(String(3), nullable=False, default="USD")
    quote_status = Column(String(30), nullable=False, default="DRAFT")
    lifecycle_status = Column(String(30), nullable=False, default="QUOTED", index=True)
    job_status = Column(String(30), nullable=True)
    invoice_status = Column(String(30), nullable=True)
    payment_status = Column(String(30), nullable=True)
    subtotal_cents = Column(Integer, nullable=False)
    tax_cents = Column(Integer, nullable=False, default=0)
    total_cents = Column(Integer, nullable=False)
    quote_expires_at = Column(DateTime(timezone=True), nullable=False)
    quote_accepted_at = Column(DateTime(timezone=True), nullable=True)
    booking_start = Column(DateTime(timezone=True), nullable=True, index=True)
    booking_end = Column(DateTime(timezone=True), nullable=True, index=True)
    resource_id = Column(String(36), ForeignKey("service_resources.id"), nullable=True, index=True)
    cancellation_reason = Column(Text, nullable=True)
    version = Column(Integer, nullable=False, default=1)
    created_by = Column(String(255), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    resource = relationship("ServiceResource")
    items = relationship("ServiceOrderItem", back_populates="order", cascade="all, delete-orphan")
    changes = relationship("ServiceChangeOrder", back_populates="order", cascade="all, delete-orphan")
    invoice = relationship("ServiceInvoice", back_populates="order", uselist=False, cascade="all, delete-orphan")
    events = relationship("ServiceOrderEvent", back_populates="order", order_by="ServiceOrderEvent.sequence")

    __table_args__ = (
        CheckConstraint("subtotal_cents >= 0 AND tax_cents >= 0 AND total_cents >= 0", name="service_order_amounts_nonnegative"),
    )


class ServiceOrderItem(Base):
    __tablename__ = "service_order_items"

    id = Column(String(36), primary_key=True, default=new_id)
    order_id = Column(String(36), ForeignKey("service_orders.id", ondelete="CASCADE"), nullable=False, index=True)
    inventory_id = Column(String(36), ForeignKey("service_inventory.id"), nullable=False)
    quantity = Column(Integer, nullable=False)
    unit_price_cents = Column(Integer, nullable=False)
    reservation_status = Column(String(30), nullable=False, default="PENDING")

    order = relationship("ServiceOrder", back_populates="items")
    inventory = relationship("InventoryItem")

    __table_args__ = (
        CheckConstraint("quantity > 0", name="service_order_item_quantity_positive"),
        CheckConstraint("unit_price_cents >= 0", name="service_order_item_price_nonnegative"),
    )


class ServiceChangeOrder(Base):
    __tablename__ = "service_change_orders"

    id = Column(String(36), primary_key=True, default=new_id)
    order_id = Column(String(36), ForeignKey("service_orders.id", ondelete="CASCADE"), nullable=False, index=True)
    idempotency_key = Column(String(200), nullable=False, unique=True)
    description = Column(Text, nullable=False)
    amount_delta_cents = Column(Integer, nullable=False)
    status = Column(String(30), nullable=False, default="PROPOSED")
    proposed_by = Column(String(255), nullable=False)
    decided_by = Column(String(255), nullable=True)
    decision_reason = Column(Text, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    decided_at = Column(DateTime(timezone=True), nullable=True)

    order = relationship("ServiceOrder", back_populates="changes")


class ServiceInvoice(Base):
    __tablename__ = "service_invoices"

    id = Column(String(36), primary_key=True, default=new_id)
    order_id = Column(String(36), ForeignKey("service_orders.id", ondelete="RESTRICT"), nullable=False, unique=True)
    subtotal_cents = Column(Integer, nullable=False)
    tax_cents = Column(Integer, nullable=False)
    total_cents = Column(Integer, nullable=False)
    amount_paid_cents = Column(Integer, nullable=False, default=0)
    amount_refunded_cents = Column(Integer, nullable=False, default=0)
    status = Column(String(30), nullable=False, default="DRAFT")
    external_accounting_id = Column(String(255), nullable=True)
    issued_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())
    updated_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now(), onupdate=func.now())

    order = relationship("ServiceOrder", back_populates="invoice")
    payments = relationship("ServicePayment", back_populates="invoice", cascade="all, delete-orphan")

    __table_args__ = (
        CheckConstraint(
            "total_cents >= 0 AND amount_paid_cents >= 0 AND amount_refunded_cents >= 0",
            name="service_invoice_amounts_nonnegative",
        ),
    )


class ServicePayment(Base):
    __tablename__ = "service_payments"

    id = Column(String(36), primary_key=True, default=new_id)
    invoice_id = Column(String(36), ForeignKey("service_invoices.id", ondelete="RESTRICT"), nullable=False, index=True)
    provider = Column(String(100), nullable=False)
    provider_event_id = Column(String(255), nullable=False)
    kind = Column(String(30), nullable=False)
    amount_cents = Column(Integer, nullable=False)
    status = Column(String(30), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    invoice = relationship("ServiceInvoice", back_populates="payments")

    __table_args__ = (
        UniqueConstraint("provider", "provider_event_id", name="uq_service_payment_provider_event"),
        CheckConstraint("amount_cents > 0", name="service_payment_amount_positive"),
    )


class ProviderEvent(Base):
    __tablename__ = "service_provider_events"

    id = Column(String(36), primary_key=True, default=new_id)
    provider = Column(String(100), nullable=False)
    event_id = Column(String(255), nullable=False)
    event_type = Column(String(100), nullable=False)
    payload_hash = Column(String(64), nullable=False)
    result = Column(JSON_TYPE, nullable=False, default=dict)
    processed_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (UniqueConstraint("provider", "event_id", name="uq_service_provider_event"),)


class CustomerMessage(Base):
    __tablename__ = "service_customer_messages"

    id = Column(String(36), primary_key=True, default=new_id)
    order_id = Column(String(36), ForeignKey("service_orders.id", ondelete="CASCADE"), nullable=False, index=True)
    idempotency_key = Column(String(200), nullable=False, unique=True)
    channel = Column(String(30), nullable=False)
    template_key = Column(String(100), nullable=False)
    payload = Column(JSON_TYPE, nullable=False, default=dict)
    status = Column(String(30), nullable=False, default="QUEUED")
    provider_message_id = Column(String(255), nullable=True)
    attempts = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())


class OfflineOperation(Base):
    __tablename__ = "service_offline_operations"

    id = Column(String(36), primary_key=True, default=new_id)
    device_id = Column(String(200), nullable=False)
    operation_id = Column(String(200), nullable=False)
    order_id = Column(String(36), ForeignKey("service_orders.id", ondelete="CASCADE"), nullable=False)
    expected_version = Column(Integer, nullable=False)
    operation = Column(String(100), nullable=False)
    payload = Column(JSON_TYPE, nullable=False, default=dict)
    status = Column(String(30), nullable=False)
    result = Column(JSON_TYPE, nullable=False, default=dict)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    __table_args__ = (UniqueConstraint("device_id", "operation_id", name="uq_service_offline_operation"),)


class ServiceOrderEvent(Base):
    __tablename__ = "service_order_events"

    id = Column(String(36), primary_key=True, default=new_id)
    order_id = Column(String(36), ForeignKey("service_orders.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    event_type = Column(String(100), nullable=False)
    actor_id = Column(String(255), nullable=False)
    payload = Column(JSON_TYPE, nullable=False, default=dict)
    previous_hash = Column(String(64), nullable=False)
    event_hash = Column(String(64), nullable=False, unique=True)
    created_at = Column(DateTime(timezone=True), nullable=False, server_default=func.now())

    order = relationship("ServiceOrder", back_populates="events")

    __table_args__ = (UniqueConstraint("order_id", "sequence", name="uq_service_order_event_sequence"),)
