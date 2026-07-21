from __future__ import annotations

import hashlib
import json
from datetime import datetime, timedelta, timezone
from typing import Any

from sqlalchemy.orm import Session

from admin.models.service_order import (
    CustomerMessage,
    InventoryItem,
    OfflineOperation,
    ProviderEvent,
    ServiceChangeOrder,
    ServiceInvoice,
    ServiceOrder,
    ServiceOrderEvent,
    ServiceOrderItem,
    ServicePayment,
    ServiceResource,
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def aware(value: datetime | None) -> datetime | None:
    if value is not None and value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value


class DomainError(Exception):
    def __init__(self, code: str, message: str, status_code: int = 409):
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


class ServiceOrderService:
    """Transactional, provider-independent service-order lifecycle."""

    ACTIVE_BOOKING_STATUSES = {"BOOKED", "ACTIVE"}

    def __init__(self, db: Session):
        self.db = db

    def _commit(self) -> None:
        try:
            self.db.commit()
        except Exception:
            self.db.rollback()
            raise

    def _order(self, order_id: str, lock: bool = False) -> ServiceOrder:
        query = self.db.query(ServiceOrder).filter(ServiceOrder.id == order_id)
        if lock:
            query = query.with_for_update()
        order = query.first()
        if not order:
            raise DomainError("order_not_found", "Service order was not found", 404)
        return order

    def _resource(self, resource_id: str, lock: bool = False) -> ServiceResource:
        query = self.db.query(ServiceResource).filter(ServiceResource.id == resource_id)
        if lock:
            query = query.with_for_update()
        resource = query.first()
        if not resource or not resource.is_active:
            raise DomainError("resource_not_found", "Active service resource was not found", 404)
        return resource

    def _event(self, order: ServiceOrder, event_type: str, actor_id: str, payload: dict[str, Any]) -> None:
        previous = (
            self.db.query(ServiceOrderEvent)
            .filter(ServiceOrderEvent.order_id == order.id)
            .order_by(ServiceOrderEvent.sequence.desc())
            .first()
        )
        sequence = (previous.sequence + 1) if previous else 1
        previous_hash = previous.event_hash if previous else "0" * 64
        created_at = utcnow()
        material = {
            "order_id": order.id,
            "sequence": sequence,
            "event_type": event_type,
            "actor_id": str(actor_id),
            "payload": payload,
            "previous_hash": previous_hash,
            "created_at": created_at.isoformat(),
        }
        event_hash = hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        self.db.add(ServiceOrderEvent(
            order_id=order.id,
            sequence=sequence,
            event_type=event_type,
            actor_id=str(actor_id),
            payload=payload,
            previous_hash=previous_hash,
            event_hash=event_hash,
            created_at=created_at,
        ))

    def _message(self, order: ServiceOrder, template: str, suffix: str, payload: dict[str, Any]) -> None:
        channel = "email" if order.customer_email else "sms" if order.customer_phone else "internal"
        key = f"{order.id}:{template}:{suffix}"
        existing = self.db.query(CustomerMessage).filter(CustomerMessage.idempotency_key == key).first()
        if not existing:
            self.db.add(CustomerMessage(
                order_id=order.id,
                idempotency_key=key,
                channel=channel,
                template_key=template,
                payload=payload,
            ))

    def _bump(self, order: ServiceOrder) -> None:
        order.version += 1

    def create_resource(
        self, *, name: str, skills: list[str], service_area_codes: list[str],
        travel_buffer_minutes: int = 30,
    ) -> ServiceResource:
        if not name.strip() or not skills or not service_area_codes:
            raise DomainError("invalid_resource", "Name, skills, and service areas are required", 422)
        if travel_buffer_minutes < 0 or travel_buffer_minutes > 240:
            raise DomainError("invalid_travel_buffer", "Travel buffer must be between 0 and 240 minutes", 422)
        resource = ServiceResource(
            name=name.strip(), skills=sorted(set(skills)),
            service_area_codes=sorted(set(service_area_codes)),
            travel_buffer_minutes=travel_buffer_minutes,
        )
        self.db.add(resource)
        self._commit()
        self.db.refresh(resource)
        return resource

    def upsert_inventory(self, *, sku: str, name: str, on_hand: int, unit_price_cents: int) -> InventoryItem:
        if not sku.strip() or on_hand < 0 or unit_price_cents < 0:
            raise DomainError("invalid_inventory", "SKU and non-negative quantities/prices are required", 422)
        item = self.db.query(InventoryItem).filter(InventoryItem.sku == sku.strip()).with_for_update().first()
        if item:
            if on_hand < item.reserved:
                raise DomainError("inventory_reserved", "On-hand quantity cannot be below reserved quantity")
            item.name, item.on_hand, item.unit_price_cents = name.strip(), on_hand, unit_price_cents
            item.version += 1
        else:
            item = InventoryItem(sku=sku.strip(), name=name.strip(), on_hand=on_hand, unit_price_cents=unit_price_cents)
            self.db.add(item)
        self._commit()
        self.db.refresh(item)
        return item

    def create_quote(
        self, *, idempotency_key: str, customer_name: str, customer_email: str | None,
        customer_phone: str | None, service_code: str, required_skills: list[str],
        service_area_code: str, item_quantities: list[dict[str, Any]], tax_basis_points: int,
        expires_at: datetime, actor_id: str,
    ) -> ServiceOrder:
        prior = self.db.query(ServiceOrder).filter(ServiceOrder.idempotency_key == idempotency_key).first()
        if prior:
            return prior
        if not idempotency_key or not customer_name.strip() or not service_code.strip() or not required_skills:
            raise DomainError("invalid_quote", "Idempotency key, customer, service, and required skills are required", 422)
        if aware(expires_at) <= utcnow():
            raise DomainError("invalid_expiry", "Quote expiry must be in the future", 422)
        if tax_basis_points < 0 or tax_basis_points > 10_000:
            raise DomainError("invalid_tax", "Tax basis points must be between 0 and 10000", 422)
        normalized: list[tuple[InventoryItem, int]] = []
        subtotal = 0
        seen: set[str] = set()
        for requested in item_quantities:
            sku, quantity = str(requested.get("sku", "")).strip(), int(requested.get("quantity", 0))
            if not sku or quantity <= 0 or sku in seen:
                raise DomainError("invalid_quote_item", "Quote items require unique SKUs and positive quantities", 422)
            seen.add(sku)
            inventory = self.db.query(InventoryItem).filter(InventoryItem.sku == sku).first()
            if not inventory:
                raise DomainError("inventory_not_found", f"Inventory SKU {sku} was not found", 404)
            normalized.append((inventory, quantity))
            subtotal += inventory.unit_price_cents * quantity
        tax = (subtotal * tax_basis_points + 5_000) // 10_000
        order = ServiceOrder(
            idempotency_key=idempotency_key,
            customer_name=customer_name.strip(), customer_email=customer_email, customer_phone=customer_phone,
            service_code=service_code.strip(), required_skills=sorted(set(required_skills)),
            service_area_code=service_area_code.strip(), subtotal_cents=subtotal, tax_cents=tax,
            total_cents=subtotal + tax, quote_expires_at=aware(expires_at), created_by=str(actor_id),
        )
        self.db.add(order)
        self.db.flush()
        for inventory, quantity in normalized:
            self.db.add(ServiceOrderItem(
                order_id=order.id, inventory_id=inventory.id, quantity=quantity,
                unit_price_cents=inventory.unit_price_cents,
            ))
        self._event(order, "QUOTE_CREATED", actor_id, {"total_cents": order.total_cents})
        self._commit()
        return order

    def send_quote(self, order_id: str, actor_id: str) -> ServiceOrder:
        order = self._order(order_id, True)
        if order.quote_status == "SENT":
            return order
        if order.quote_status != "DRAFT":
            raise DomainError("invalid_quote_state", "Only draft quotes can be sent")
        if aware(order.quote_expires_at) <= utcnow():
            order.quote_status = "EXPIRED"
            self._commit()
            raise DomainError("quote_expired", "Quote has expired")
        order.quote_status = "SENT"
        self._bump(order)
        self._event(order, "QUOTE_SENT", actor_id, {})
        self._message(order, "quote_sent", str(order.version), {"total_cents": order.total_cents})
        self._commit()
        return order

    def accept_quote(self, order_id: str, actor_id: str) -> ServiceOrder:
        order = self._order(order_id, True)
        if order.quote_status == "ACCEPTED":
            return order
        if order.quote_status != "SENT":
            raise DomainError("invalid_quote_state", "Only sent quotes can be accepted")
        if aware(order.quote_expires_at) <= utcnow():
            order.quote_status = "EXPIRED"
            self._commit()
            raise DomainError("quote_expired", "Quote has expired")
        order.quote_status, order.quote_accepted_at = "ACCEPTED", utcnow()
        self._bump(order)
        self._event(order, "QUOTE_ACCEPTED", actor_id, {})
        self._commit()
        return order

    def availability(
        self, *, required_skills: list[str], service_area_code: str,
        start: datetime, end: datetime, exclude_order_id: str | None = None,
    ) -> list[ServiceResource]:
        start, end = aware(start), aware(end)
        if start is None or end is None or end <= start:
            raise DomainError("invalid_window", "Booking end must be after start", 422)
        resources = self.db.query(ServiceResource).filter(ServiceResource.is_active.is_(True)).all()
        bookings = self.db.query(ServiceOrder).filter(
            ServiceOrder.resource_id.is_not(None), ServiceOrder.lifecycle_status.in_(self.ACTIVE_BOOKING_STATUSES)
        ).all()
        available = []
        required = set(required_skills)
        for resource in resources:
            if not required.issubset(set(resource.skills or [])) or service_area_code not in (resource.service_area_codes or []):
                continue
            buffer = timedelta(minutes=resource.travel_buffer_minutes)
            conflict = any(
                booking.id != exclude_order_id
                and booking.resource_id == resource.id
                and aware(booking.booking_start) < end + buffer
                and aware(booking.booking_end) > start - buffer
                for booking in bookings
            )
            if not conflict:
                available.append(resource)
        return available

    def _reserve_items(self, order: ServiceOrder) -> None:
        for line in order.items:
            inventory = self.db.query(InventoryItem).filter(InventoryItem.id == line.inventory_id).with_for_update().one()
            if inventory.on_hand - inventory.reserved < line.quantity:
                raise DomainError("insufficient_inventory", f"Insufficient available inventory for {inventory.sku}")
            inventory.reserved += line.quantity
            inventory.version += 1
            line.reservation_status = "RESERVED"

    def _release_items(self, order: ServiceOrder, consume: bool = False) -> None:
        for line in order.items:
            if line.reservation_status != "RESERVED":
                continue
            inventory = self.db.query(InventoryItem).filter(InventoryItem.id == line.inventory_id).with_for_update().one()
            inventory.reserved -= line.quantity
            if consume:
                inventory.on_hand -= line.quantity
                line.reservation_status = "CONSUMED"
            else:
                line.reservation_status = "RELEASED"
            inventory.version += 1

    def book(self, order_id: str, *, start: datetime, end: datetime, actor_id: str, resource_id: str | None = None) -> ServiceOrder:
        order = self._order(order_id, True)
        if order.lifecycle_status == "BOOKED" and aware(order.booking_start) == aware(start) and aware(order.booking_end) == aware(end):
            return order
        if order.quote_status != "ACCEPTED" or order.lifecycle_status != "QUOTED":
            raise DomainError("invalid_booking_state", "An accepted, unbooked quote is required")
        candidates = self.availability(
            required_skills=order.required_skills, service_area_code=order.service_area_code,
            start=start, end=end, exclude_order_id=order.id,
        )
        candidate_ids = [candidate.id for candidate in candidates]
        if resource_id:
            self._resource(resource_id)
            candidate_ids = [candidate_id for candidate_id in candidate_ids if candidate_id == resource_id]
        selected = None
        # Lock and recheck the resource row so concurrent dispatchers cannot both
        # pass the availability read and create overlapping bookings.
        for candidate_id in candidate_ids:
            candidate = self._resource(candidate_id, lock=True)
            rechecked = self.availability(
                required_skills=order.required_skills, service_area_code=order.service_area_code,
                start=start, end=end, exclude_order_id=order.id,
            )
            if any(item.id == candidate.id for item in rechecked):
                selected = candidate
                break
        if not selected:
            raise DomainError("no_availability", "No qualified resource is available for this window")
        self._reserve_items(order)
        order.resource_id, order.booking_start, order.booking_end = selected.id, aware(start), aware(end)
        order.lifecycle_status, order.job_status = "BOOKED", "SCHEDULED"
        self._bump(order)
        self._event(order, "BOOKED", actor_id, {"resource_id": order.resource_id, "start": aware(start).isoformat(), "end": aware(end).isoformat()})
        self._message(order, "booking_confirmed", str(order.version), {"start": aware(start).isoformat()})
        self._commit()
        return order

    def reschedule(self, order_id: str, *, start: datetime, end: datetime, actor_id: str) -> ServiceOrder:
        order = self._order(order_id, True)
        if order.lifecycle_status != "BOOKED" or order.job_status != "SCHEDULED":
            raise DomainError("invalid_reschedule_state", "Only scheduled bookings can be rescheduled")
        self._resource(order.resource_id, lock=True)
        candidates = self.availability(
            required_skills=order.required_skills, service_area_code=order.service_area_code,
            start=start, end=end, exclude_order_id=order.id,
        )
        if not any(resource.id == order.resource_id for resource in candidates):
            raise DomainError("no_availability", "Assigned resource is unavailable for the requested window")
        old_start = aware(order.booking_start).isoformat()
        order.booking_start, order.booking_end = aware(start), aware(end)
        self._bump(order)
        self._event(order, "RESCHEDULED", actor_id, {"old_start": old_start, "new_start": aware(start).isoformat()})
        self._message(order, "booking_rescheduled", str(order.version), {"start": aware(start).isoformat()})
        self._commit()
        return order

    def reassign(self, order_id: str, *, resource_id: str, actor_id: str) -> ServiceOrder:
        order = self._order(order_id, True)
        if order.lifecycle_status != "BOOKED" or order.job_status != "SCHEDULED":
            raise DomainError("invalid_reassign_state", "Only scheduled bookings can be reassigned")
        self._resource(resource_id, lock=True)
        candidates = self.availability(
            required_skills=order.required_skills, service_area_code=order.service_area_code,
            start=order.booking_start, end=order.booking_end, exclude_order_id=order.id,
        )
        if not any(resource.id == resource_id for resource in candidates):
            raise DomainError("resource_unavailable", "Resource is unqualified or unavailable")
        previous = order.resource_id
        order.resource_id = resource_id
        self._bump(order)
        self._event(order, "REASSIGNED", actor_id, {"old_resource_id": previous, "resource_id": resource_id})
        self._commit()
        return order

    def dispatch(self, order_id: str, actor_id: str) -> ServiceOrder:
        order = self._order(order_id, True)
        if order.job_status == "DISPATCHED":
            return order
        if order.lifecycle_status != "BOOKED" or order.job_status != "SCHEDULED":
            raise DomainError("invalid_dispatch_state", "Only scheduled bookings can be dispatched")
        order.lifecycle_status, order.job_status = "ACTIVE", "DISPATCHED"
        self._bump(order)
        self._event(order, "DISPATCHED", actor_id, {"resource_id": order.resource_id})
        self._message(order, "technician_dispatched", str(order.version), {})
        self._commit()
        return order

    def update_job_status(self, order_id: str, status: str, actor_id: str) -> ServiceOrder:
        order = self._order(order_id, True)
        status = status.upper()
        if order.job_status == status:
            return order
        allowed = {
            "DISPATCHED": {"EN_ROUTE", "IN_PROGRESS", "CANCELLED"},
            "EN_ROUTE": {"IN_PROGRESS", "NO_SHOW", "CANCELLED"},
            "IN_PROGRESS": {"PARTIAL", "COMPLETED"},
            "PARTIAL": {"IN_PROGRESS", "COMPLETED"},
        }
        if status not in allowed.get(order.job_status or "", set()):
            raise DomainError("invalid_job_transition", f"Cannot move job from {order.job_status} to {status}")
        order.job_status = status
        if status == "NO_SHOW":
            self._release_items(order)
            order.lifecycle_status = "NO_SHOW"
        elif status == "CANCELLED":
            self._release_items(order)
            order.lifecycle_status = "CANCELLED"
        elif status == "COMPLETED":
            self._release_items(order, consume=True)
            order.lifecycle_status = "COMPLETED"
        self._bump(order)
        self._event(order, f"JOB_{status}", actor_id, {})
        if status in {"NO_SHOW", "PARTIAL", "COMPLETED", "CANCELLED"}:
            self._message(order, f"job_{status.lower()}", str(order.version), {})
        self._commit()
        return order

    def propose_change(self, order_id: str, *, idempotency_key: str, description: str, amount_delta_cents: int, actor_id: str) -> ServiceChangeOrder:
        prior = self.db.query(ServiceChangeOrder).filter(ServiceChangeOrder.idempotency_key == idempotency_key).first()
        if prior:
            return prior
        order = self._order(order_id, True)
        if order.lifecycle_status not in {"BOOKED", "ACTIVE"}:
            raise DomainError("invalid_change_state", "Changes require a booked or active order")
        if not description.strip() or order.total_cents + amount_delta_cents < 0:
            raise DomainError("invalid_change", "Description is required and revised total cannot be negative", 422)
        change = ServiceChangeOrder(
            order_id=order.id, idempotency_key=idempotency_key, description=description.strip(),
            amount_delta_cents=amount_delta_cents, proposed_by=str(actor_id),
        )
        self.db.add(change)
        self.db.flush()
        self._event(order, "CHANGE_PROPOSED", actor_id, {"change_id": change.id, "amount_delta_cents": amount_delta_cents})
        self._commit()
        return change

    def decide_change(self, order_id: str, change_id: str, *, approve: bool, reason: str, actor_id: str) -> ServiceChangeOrder:
        order = self._order(order_id, True)
        change = self.db.query(ServiceChangeOrder).filter(
            ServiceChangeOrder.id == change_id, ServiceChangeOrder.order_id == order.id
        ).with_for_update().first()
        if not change:
            raise DomainError("change_not_found", "Change order was not found", 404)
        if change.status != "PROPOSED":
            return change
        if str(actor_id) == change.proposed_by:
            raise DomainError("separation_of_duties", "The proposer cannot approve or reject this change", 403)
        if not reason.strip():
            raise DomainError("decision_reason_required", "A decision reason is required", 422)
        change.status, change.decided_by, change.decision_reason, change.decided_at = (
            "APPROVED" if approve else "REJECTED", str(actor_id), reason.strip(), utcnow()
        )
        if approve:
            order.subtotal_cents += change.amount_delta_cents
            order.total_cents += change.amount_delta_cents
            self._bump(order)
        self._event(order, f"CHANGE_{change.status}", actor_id, {"change_id": change.id, "reason": reason.strip()})
        self._commit()
        return change

    def issue_invoice(self, order_id: str, actor_id: str) -> ServiceInvoice:
        order = self._order(order_id, True)
        if order.invoice:
            return order.invoice
        if order.job_status not in {"PARTIAL", "COMPLETED"}:
            raise DomainError("invalid_invoice_state", "Invoice requires partial or completed work")
        invoice = ServiceInvoice(
            order_id=order.id, subtotal_cents=order.subtotal_cents, tax_cents=order.tax_cents,
            total_cents=order.total_cents, status="ISSUED", issued_at=utcnow(),
        )
        order.invoice_status, order.payment_status = "ISSUED", "UNPAID"
        self.db.add(invoice)
        self.db.flush()
        self._bump(order)
        self._event(order, "INVOICE_ISSUED", actor_id, {"invoice_id": invoice.id, "total_cents": invoice.total_cents})
        self._message(order, "invoice_issued", str(order.version), {"invoice_id": invoice.id})
        self._commit()
        return invoice

    def process_payment_event(
        self, *, provider: str, event_id: str, event_type: str, order_id: str,
        amount_cents: int, payload_hash: str,
    ) -> dict[str, Any]:
        prior = self.db.query(ProviderEvent).filter(
            ProviderEvent.provider == provider, ProviderEvent.event_id == event_id
        ).first()
        if prior:
            if prior.payload_hash != payload_hash:
                raise DomainError("webhook_replay_mismatch", "Event ID was reused with a different payload")
            return prior.result
        if amount_cents <= 0:
            raise DomainError("invalid_payment_amount", "Payment amount must be positive", 422)
        order = self._order(order_id, True)
        # The order lock serializes event processing. Recheck after acquiring it
        # because another request may have committed while this request waited.
        prior = self.db.query(ProviderEvent).filter(
            ProviderEvent.provider == provider, ProviderEvent.event_id == event_id
        ).first()
        if prior:
            if prior.payload_hash != payload_hash:
                raise DomainError("webhook_replay_mismatch", "Event ID was reused with a different payload")
            return prior.result
        invoice = self.db.query(ServiceInvoice).filter(ServiceInvoice.order_id == order.id).with_for_update().first()
        if not invoice or invoice.status not in {"ISSUED", "PARTIALLY_PAID", "PAID", "PARTIALLY_REFUNDED"}:
            raise DomainError("invoice_not_payable", "Order does not have a payable invoice")
        event_type = event_type.upper()
        if event_type == "CHARGE_SUCCEEDED":
            if invoice.amount_paid_cents + amount_cents > invoice.total_cents:
                raise DomainError("payment_exceeds_balance", "Payment exceeds the invoice balance")
            invoice.amount_paid_cents += amount_cents
            kind = "CHARGE"
        elif event_type == "REFUND_SUCCEEDED":
            if invoice.amount_refunded_cents + amount_cents > invoice.amount_paid_cents:
                raise DomainError("refund_exceeds_payment", "Refund exceeds captured payments")
            invoice.amount_refunded_cents += amount_cents
            kind = "REFUND"
        else:
            raise DomainError("unsupported_payment_event", "Unsupported payment event type", 422)
        net_paid = invoice.amount_paid_cents - invoice.amount_refunded_cents
        if invoice.amount_refunded_cents == invoice.amount_paid_cents and invoice.amount_paid_cents:
            invoice.status = order.payment_status = "REFUNDED"
        elif invoice.amount_refunded_cents:
            invoice.status = order.payment_status = "PARTIALLY_REFUNDED"
        elif net_paid == invoice.total_cents:
            invoice.status = order.payment_status = "PAID"
        else:
            invoice.status = order.payment_status = "PARTIALLY_PAID"
        payment = ServicePayment(
            invoice_id=invoice.id, provider=provider, provider_event_id=event_id,
            kind=kind, amount_cents=amount_cents, status="SUCCEEDED",
        )
        self.db.add(payment)
        self.db.flush()
        result = {"order_id": order.id, "invoice_id": invoice.id, "payment_status": order.payment_status}
        self.db.add(ProviderEvent(
            provider=provider, event_id=event_id, event_type=event_type,
            payload_hash=payload_hash, result=result,
        ))
        self._bump(order)
        self._event(order, f"PAYMENT_{kind}_SUCCEEDED", f"provider:{provider}", {"amount_cents": amount_cents, "event_id": event_id})
        self._commit()
        return result

    def cancel(self, order_id: str, *, reason: str, actor_id: str) -> ServiceOrder:
        order = self._order(order_id, True)
        if order.lifecycle_status == "CANCELLED":
            return order
        if order.lifecycle_status not in {"QUOTED", "BOOKED"}:
            raise DomainError("invalid_cancellation_state", "Only quoted or scheduled orders can be cancelled")
        if order.payment_status not in {None, "UNPAID", "REFUNDED"}:
            raise DomainError("refund_required", "Captured funds must be refunded before cancellation")
        if not reason.strip():
            raise DomainError("cancellation_reason_required", "A cancellation reason is required", 422)
        self._release_items(order)
        order.lifecycle_status, order.job_status, order.cancellation_reason = "CANCELLED", "CANCELLED", reason.strip()
        self._bump(order)
        self._event(order, "CANCELLED", actor_id, {"reason": reason.strip()})
        self._message(order, "order_cancelled", str(order.version), {"reason": reason.strip()})
        self._commit()
        return order

    def replay_offline(
        self, *, device_id: str, operation_id: str, order_id: str,
        expected_version: int, operation: str, payload: dict[str, Any], actor_id: str,
    ) -> dict[str, Any]:
        prior = self.db.query(OfflineOperation).filter(
            OfflineOperation.device_id == device_id, OfflineOperation.operation_id == operation_id
        ).first()
        if prior:
            return prior.result
        order = self._order(order_id, True)
        if order.version != expected_version:
            result = {"status": "CONFLICT", "current_version": order.version}
            self.db.add(OfflineOperation(
                device_id=device_id, operation_id=operation_id, order_id=order.id,
                expected_version=expected_version, operation=operation, payload=payload,
                status="CONFLICT", result=result,
            ))
            self._commit()
            return result
        if operation != "job_status":
            raise DomainError("unsupported_offline_operation", "Only job_status can be replayed offline", 422)
        # Record the receipt first; update_job_status commits the combined unit of work.
        result = {"status": "APPLIED", "order_id": order.id, "requested_status": payload.get("status")}
        self.db.add(OfflineOperation(
            device_id=device_id, operation_id=operation_id, order_id=order.id,
            expected_version=expected_version, operation=operation, payload=payload,
            status="APPLIED", result=result,
        ))
        updated = self.update_job_status(order.id, str(payload.get("status", "")), actor_id)
        result["current_version"] = updated.version
        record = self.db.query(OfflineOperation).filter(
            OfflineOperation.device_id == device_id, OfflineOperation.operation_id == operation_id
        ).one()
        record.result = result
        self._commit()
        return result

    def verify_event_chain(self, order_id: str) -> bool:
        events = self.db.query(ServiceOrderEvent).filter(
            ServiceOrderEvent.order_id == order_id
        ).order_by(ServiceOrderEvent.sequence).all()
        previous_hash = "0" * 64
        for expected_sequence, event in enumerate(events, 1):
            material = {
                "order_id": event.order_id,
                "sequence": event.sequence,
                "event_type": event.event_type,
                "actor_id": event.actor_id,
                "payload": event.payload,
                "previous_hash": event.previous_hash,
                "created_at": aware(event.created_at).isoformat(),
            }
            calculated = hashlib.sha256(
                json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
            ).hexdigest()
            if event.sequence != expected_sequence or event.previous_hash != previous_hash or event.event_hash != calculated:
                return False
            previous_hash = event.event_hash
        return bool(events)

    def snapshot(self, order_id: str) -> dict[str, Any]:
        order = self._order(order_id)
        return {
            "id": order.id,
            "version": order.version,
            "customer_name": order.customer_name,
            "service_code": order.service_code,
            "quote_status": order.quote_status,
            "lifecycle_status": order.lifecycle_status,
            "job_status": order.job_status,
            "invoice_status": order.invoice_status,
            "payment_status": order.payment_status,
            "subtotal_cents": order.subtotal_cents,
            "tax_cents": order.tax_cents,
            "total_cents": order.total_cents,
            "booking_start": aware(order.booking_start).isoformat() if order.booking_start else None,
            "booking_end": aware(order.booking_end).isoformat() if order.booking_end else None,
            "resource_id": order.resource_id,
            "event_chain_valid": self.verify_event_chain(order.id),
            "events": [
                {"sequence": event.sequence, "type": event.event_type, "actor_id": event.actor_id, "payload": event.payload}
                for event in order.events
            ],
        }
