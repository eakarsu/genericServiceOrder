from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from admin.config import PROVIDER_WEBHOOK_SECRET
from admin.database import get_db
from admin.dependencies import get_current_user, require_role
from admin.models.user import User
from admin.schemas.service_order import (
    BookingRequest,
    CancellationRequest,
    ChangeCreate,
    ChangeDecision,
    InventoryUpsert,
    JobStatusRequest,
    OfflineReplayRequest,
    QuoteCreate,
    ReassignRequest,
    ResourceCreate,
)
from admin.services.provider_contracts import decode_payment_event, payload_digest, verify_webhook_signature
from admin.services.service_order_service import DomainError, ServiceOrderService

router = APIRouter()


def service_actor(current_user: User = Depends(get_current_user)) -> User:
    return current_user


def call(operation):
    try:
        return operation()
    except DomainError as exc:
        raise HTTPException(status_code=exc.status_code, detail={"code": exc.code, "message": exc.message}) from exc


@router.post("/resources")
def create_resource(body: ResourceCreate, db: Session = Depends(get_db), user: User = Depends(require_role("admin", "manager"))):
    resource = call(lambda: ServiceOrderService(db).create_resource(**body.model_dump()))
    return {"id": resource.id, "name": resource.name, "skills": resource.skills, "service_area_codes": resource.service_area_codes}


@router.put("/inventory")
def upsert_inventory(body: InventoryUpsert, db: Session = Depends(get_db), user: User = Depends(require_role("admin", "manager"))):
    item = call(lambda: ServiceOrderService(db).upsert_inventory(**body.model_dump()))
    return {"id": item.id, "sku": item.sku, "on_hand": item.on_hand, "reserved": item.reserved, "version": item.version}


@router.post("/quotes")
def create_quote(body: QuoteCreate, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    values = body.model_dump()
    values["item_quantities"] = values.pop("items")
    order = call(lambda: ServiceOrderService(db).create_quote(**values, actor_id=str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/{order_id}/quote/send")
def send_quote(order_id: str, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).send_quote(order_id, str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/{order_id}/quote/accept")
def accept_quote(order_id: str, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).accept_quote(order_id, str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.get("/availability")
def availability(required_skill: list[str], service_area_code: str, start: str, end: str, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    from datetime import datetime
    try:
        parsed_start, parsed_end = datetime.fromisoformat(start), datetime.fromisoformat(end)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail="start and end must be ISO-8601 timestamps") from exc
    resources = call(lambda: ServiceOrderService(db).availability(
        required_skills=required_skill, service_area_code=service_area_code, start=parsed_start, end=parsed_end
    ))
    return [{"id": resource.id, "name": resource.name} for resource in resources]


@router.post("/{order_id}/book")
def book(order_id: str, body: BookingRequest, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).book(order_id, **body.model_dump(), actor_id=str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/{order_id}/reschedule")
def reschedule(order_id: str, body: BookingRequest, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).reschedule(order_id, start=body.start, end=body.end, actor_id=str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/{order_id}/reassign")
def reassign(order_id: str, body: ReassignRequest, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).reassign(order_id, resource_id=body.resource_id, actor_id=str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/{order_id}/dispatch")
def dispatch(order_id: str, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).dispatch(order_id, str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/{order_id}/job-status")
def job_status(order_id: str, body: JobStatusRequest, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).update_job_status(order_id, body.status, str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/{order_id}/changes")
def propose_change(order_id: str, body: ChangeCreate, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    change = call(lambda: ServiceOrderService(db).propose_change(order_id, **body.model_dump(), actor_id=str(user.id)))
    return {"id": change.id, "status": change.status, "amount_delta_cents": change.amount_delta_cents}


@router.post("/{order_id}/changes/{change_id}/decision")
def decide_change(order_id: str, change_id: str, body: ChangeDecision, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    change = call(lambda: ServiceOrderService(db).decide_change(order_id, change_id, **body.model_dump(), actor_id=str(user.id)))
    return {"id": change.id, "status": change.status, "decision_reason": change.decision_reason}


@router.post("/{order_id}/invoice")
def issue_invoice(order_id: str, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    invoice = call(lambda: ServiceOrderService(db).issue_invoice(order_id, str(user.id)))
    return {"id": invoice.id, "status": invoice.status, "total_cents": invoice.total_cents}


@router.post("/{order_id}/cancel")
def cancel(order_id: str, body: CancellationRequest, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    order = call(lambda: ServiceOrderService(db).cancel(order_id, reason=body.reason, actor_id=str(user.id)))
    return ServiceOrderService(db).snapshot(order.id)


@router.post("/offline/replay")
def replay(body: OfflineReplayRequest, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    return call(lambda: ServiceOrderService(db).replay_offline(**body.model_dump(), actor_id=str(user.id)))


@router.get("/{order_id}")
def get_order(order_id: str, db: Session = Depends(get_db), user: User = Depends(service_actor)):
    return call(lambda: ServiceOrderService(db).snapshot(order_id))


@router.post("/webhooks/payment/{provider}")
async def payment_webhook(provider: str, request: Request, db: Session = Depends(get_db)):
    raw = await request.body()
    if not verify_webhook_signature(raw, request.headers.get("x-webhook-signature"), PROVIDER_WEBHOOK_SECRET):
        raise HTTPException(status_code=401, detail="Invalid webhook signature")
    try:
        event = decode_payment_event(raw)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return call(lambda: ServiceOrderService(db).process_payment_event(
        provider=provider, event_id=str(event["event_id"]), event_type=str(event["event_type"]),
        order_id=str(event["order_id"]), amount_cents=int(event["amount_cents"]), payload_hash=payload_digest(raw),
    ))
