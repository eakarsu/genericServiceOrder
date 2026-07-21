from datetime import timedelta

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import admin.models  # noqa: F401
from admin.database import Base
from admin.models.service_order import InventoryItem, ServiceOrderEvent
from admin.services.service_order_service import DomainError, ServiceOrderService, utcnow


@pytest.fixture()
def db():
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine, expire_on_commit=False)()
    try:
        yield session
    finally:
        session.close()
        engine.dispose()


@pytest.fixture()
def service(db):
    return ServiceOrderService(db)


def setup_catalog(service):
    first = service.create_resource(
        name="Alex", skills=["plumbing"], service_area_codes=["10001"], travel_buffer_minutes=30
    )
    second = service.create_resource(
        name="Blair", skills=["plumbing", "electrical"], service_area_codes=["10001"], travel_buffer_minutes=30
    )
    service.upsert_inventory(sku="VALVE", name="Valve", on_hand=4, unit_price_cents=2500)
    return first, second


def quote(service, key, quantity=1):
    order = service.create_quote(
        idempotency_key=key, customer_name="Customer", customer_email="customer@example.test",
        customer_phone=None, service_code="repair", required_skills=["plumbing"],
        service_area_code="10001", item_quantities=[{"sku": "VALVE", "quantity": quantity}],
        tax_basis_points=800, expires_at=utcnow() + timedelta(days=2), actor_id="dispatcher",
    )
    service.send_quote(order.id, "dispatcher")
    service.accept_quote(order.id, "customer")
    return order


def test_complete_journey_is_audited_and_consumes_inventory(service, db):
    resource, _ = setup_catalog(service)
    order = quote(service, "complete-journey")
    start = utcnow() + timedelta(days=1)
    service.book(order.id, start=start, end=start + timedelta(hours=2), resource_id=resource.id, actor_id="dispatcher")
    service.dispatch(order.id, "dispatcher")
    service.update_job_status(order.id, "IN_PROGRESS", "technician")
    service.update_job_status(order.id, "COMPLETED", "technician")
    invoice = service.issue_invoice(order.id, "billing")
    result = service.process_payment_event(
        provider="testpay", event_id="evt-charge-1", event_type="CHARGE_SUCCEEDED",
        order_id=order.id, amount_cents=invoice.total_cents, payload_hash="a" * 64,
    )

    assert result["payment_status"] == "PAID"
    assert db.query(InventoryItem).filter_by(sku="VALVE").one().on_hand == 3
    assert service.snapshot(order.id)["event_chain_valid"] is True


def test_overbooking_and_travel_buffer_are_rejected(service):
    resource, _ = setup_catalog(service)
    first, second = quote(service, "booking-first"), quote(service, "booking-second")
    start = utcnow() + timedelta(days=1)
    service.book(first.id, start=start, end=start + timedelta(hours=1), resource_id=resource.id, actor_id="d")
    with pytest.raises(DomainError, match="No qualified resource"):
        service.book(
            second.id, start=start + timedelta(hours=1, minutes=10),
            end=start + timedelta(hours=2), resource_id=resource.id, actor_id="d",
        )


def test_no_show_releases_reserved_inventory(service, db):
    resource, _ = setup_catalog(service)
    order = quote(service, "no-show")
    start = utcnow() + timedelta(days=1)
    service.book(order.id, start=start, end=start + timedelta(hours=1), resource_id=resource.id, actor_id="d")
    assert db.query(InventoryItem).filter_by(sku="VALVE").one().reserved == 1
    service.dispatch(order.id, "d")
    service.update_job_status(order.id, "EN_ROUTE", "tech")
    service.update_job_status(order.id, "NO_SHOW", "tech")
    inventory = db.query(InventoryItem).filter_by(sku="VALVE").one()
    assert (inventory.on_hand, inventory.reserved) == (4, 0)


def test_partial_work_payment_refund_and_webhook_idempotency(service):
    resource, _ = setup_catalog(service)
    order = quote(service, "partial-refund")
    start = utcnow() + timedelta(days=1)
    service.book(order.id, start=start, end=start + timedelta(hours=1), resource_id=resource.id, actor_id="d")
    service.dispatch(order.id, "d")
    service.update_job_status(order.id, "IN_PROGRESS", "tech")
    service.update_job_status(order.id, "PARTIAL", "tech")
    invoice = service.issue_invoice(order.id, "billing")
    args = dict(provider="testpay", event_id="charge", event_type="CHARGE_SUCCEEDED", order_id=order.id,
                amount_cents=invoice.total_cents, payload_hash="1" * 64)
    assert service.process_payment_event(**args) == service.process_payment_event(**args)
    service.process_payment_event(
        provider="testpay", event_id="refund-1", event_type="REFUND_SUCCEEDED",
        order_id=order.id, amount_cents=1000, payload_hash="2" * 64,
    )
    assert service.snapshot(order.id)["payment_status"] == "PARTIALLY_REFUNDED"
    with pytest.raises(DomainError, match="different payload"):
        service.process_payment_event(**{**args, "payload_hash": "9" * 64})


def test_reschedule_reassign_change_approval_and_offline_conflict(service):
    first, second = setup_catalog(service)
    order = quote(service, "field-changes")
    start = utcnow() + timedelta(days=1)
    service.book(order.id, start=start, end=start + timedelta(hours=1), resource_id=first.id, actor_id="d")
    moved = start + timedelta(hours=3)
    service.reschedule(order.id, start=moved, end=moved + timedelta(hours=1), actor_id="d")
    service.reassign(order.id, resource_id=second.id, actor_id="d")
    change = service.propose_change(
        order.id, idempotency_key="change-0001", description="Additional fitting",
        amount_delta_cents=500, actor_id="tech",
    )
    with pytest.raises(DomainError, match="proposer"):
        service.decide_change(order.id, change.id, approve=True, reason="Approved", actor_id="tech")
    service.decide_change(order.id, change.id, approve=True, reason="Customer approved", actor_id="customer")
    conflict = service.replay_offline(
        device_id="device", operation_id="op-conflict", order_id=order.id,
        expected_version=1, operation="job_status", payload={"status": "IN_PROGRESS"}, actor_id="tech",
    )
    assert conflict["status"] == "CONFLICT"


def test_event_chain_detects_tampering(service, db):
    setup_catalog(service)
    order = quote(service, "tamper-test")
    event = db.query(ServiceOrderEvent).filter_by(order_id=order.id).first()
    event.payload = {"tampered": True}
    db.commit()
    assert service.verify_event_chain(order.id) is False
