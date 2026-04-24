import math
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import desc, asc
from sqlalchemy.orm import Session

from admin.database import get_db
from admin.models.order import Order
from admin.models.audit_log import AuditLog
from admin.models.user import User
from admin.schemas.order import (
    OrderResponse, OrderListResponse, OrderUpdate,
    BulkDeleteRequest, BulkUpdateRequest, BulkResponse,
)
from admin.dependencies import get_current_user, require_role

router = APIRouter()

VALID_SORT_FIELDS = {"id", "customer_name", "sector", "status", "total_amount", "created_at"}
VALID_STATUSES = {"pending", "confirmed", "in_progress", "completed", "cancelled"}


@router.get("", response_model=OrderListResponse)
def list_orders(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = None,
    status: str | None = None,
    sector: str | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
    sort_by: str = "created_at",
    sort_order: str = "desc",
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Order)

    if search:
        like = f"%{search}%"
        query = query.filter(
            (Order.customer_name.ilike(like))
            | (Order.customer_email.ilike(like))
            | (Order.customer_phone.ilike(like))
            | (Order.notes.ilike(like))
        )

    if status:
        query = query.filter(Order.status == status)
    if sector:
        query = query.filter(Order.sector == sector)
    if date_from:
        query = query.filter(Order.created_at >= date_from)
    if date_to:
        query = query.filter(Order.created_at <= date_to)

    total = query.count()

    if sort_by not in VALID_SORT_FIELDS:
        sort_by = "created_at"
    sort_col = getattr(Order, sort_by)
    query = query.order_by(desc(sort_col) if sort_order == "desc" else asc(sort_col))

    orders = query.offset((page - 1) * page_size).limit(page_size).all()

    return {
        "orders": orders,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
    }


@router.get("/{order_id}", response_model=OrderResponse)
def get_order(order_id: int, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")
    return order


@router.put("/{order_id}", response_model=OrderResponse)
def update_order(
    order_id: int,
    req: OrderUpdate,
    current_user: User = Depends(require_role("admin", "manager")),
    db: Session = Depends(get_db),
):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(order, key, value)

    db.add(AuditLog(user_id=current_user.id, action="update", entity_type="order", entity_id=order_id, details=update_data))
    db.commit()
    db.refresh(order)
    return order


@router.delete("/{order_id}")
def delete_order(
    order_id: int,
    current_user: User = Depends(require_role("admin", "manager")),
    db: Session = Depends(get_db),
):
    order = db.query(Order).filter(Order.id == order_id).first()
    if not order:
        raise HTTPException(status_code=404, detail="Order not found")

    db.add(AuditLog(user_id=current_user.id, action="delete", entity_type="order", entity_id=order_id))
    db.delete(order)
    db.commit()
    return {"message": "Order deleted"}


@router.post("/bulk-delete", response_model=BulkResponse)
def bulk_delete(
    req: BulkDeleteRequest,
    current_user: User = Depends(require_role("admin", "manager")),
    db: Session = Depends(get_db),
):
    count = db.query(Order).filter(Order.id.in_(req.ids)).delete(synchronize_session="fetch")
    db.add(AuditLog(user_id=current_user.id, action="bulk_delete", entity_type="order", details={"ids": req.ids}))
    db.commit()
    return {"message": f"Deleted {count} orders", "affected": count}


@router.put("/bulk-update", response_model=BulkResponse)
def bulk_update(
    req: BulkUpdateRequest,
    current_user: User = Depends(require_role("admin", "manager")),
    db: Session = Depends(get_db),
):
    count = db.query(Order).filter(Order.id.in_(req.ids)).update({"status": req.status}, synchronize_session="fetch")
    db.add(AuditLog(user_id=current_user.id, action="bulk_update", entity_type="order", details={"ids": req.ids, "status": req.status}))
    db.commit()
    return {"message": f"Updated {count} orders", "affected": count}
