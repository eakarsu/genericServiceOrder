from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from admin.database import get_db
from admin.models.sector import Sector
from admin.models.order import Order
from admin.models.user import User
from admin.schemas.sector import SectorResponse, SectorListResponse
from admin.dependencies import get_current_user

router = APIRouter()


@router.get("", response_model=SectorListResponse)
def list_sectors(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sectors = db.query(Sector).order_by(Sector.display_name).all()
    return {"sectors": sectors, "total": len(sectors)}


@router.get("/{name}")
def get_sector(name: str, current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    sector = db.query(Sector).filter(Sector.name == name).first()
    if not sector:
        raise HTTPException(status_code=404, detail="Sector not found")

    order_count = db.query(Order).filter(Order.sector == name).count()
    recent_orders = (
        db.query(Order)
        .filter(Order.sector == name)
        .order_by(Order.created_at.desc())
        .limit(10)
        .all()
    )

    return {
        "sector": SectorResponse.model_validate(sector),
        "order_count": order_count,
        "recent_orders": recent_orders,
    }
