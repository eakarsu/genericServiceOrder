from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import io

from admin.database import get_db
from admin.models.order import Order
from admin.models.user import User
from admin.schemas.export import ExportRequest
from admin.services.export_service import generate_csv, generate_pdf
from admin.dependencies import get_current_user

router = APIRouter()


def _filter_orders(db: Session, req: ExportRequest) -> list[Order]:
    query = db.query(Order)
    if req.sector:
        query = query.filter(Order.sector == req.sector)
    if req.status:
        query = query.filter(Order.status == req.status)
    if req.date_from:
        query = query.filter(Order.created_at >= req.date_from)
    if req.date_to:
        query = query.filter(Order.created_at <= req.date_to)
    if req.search:
        like = f"%{req.search}%"
        query = query.filter(
            (Order.customer_name.ilike(like))
            | (Order.customer_email.ilike(like))
        )
    return query.order_by(Order.created_at.desc()).all()


@router.post("/csv")
def export_csv(
    req: ExportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    orders = _filter_orders(db, req)
    csv_content = generate_csv(orders)
    return StreamingResponse(
        io.StringIO(csv_content),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=orders.csv"},
    )


@router.post("/pdf")
def export_pdf(
    req: ExportRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    orders = _filter_orders(db, req)
    pdf_bytes = generate_pdf(orders)
    return StreamingResponse(
        io.BytesIO(pdf_bytes),
        media_type="application/pdf",
        headers={"Content-Disposition": "attachment; filename=orders.pdf"},
    )
