from datetime import datetime, timezone, timedelta
from fastapi import APIRouter, Depends
from sqlalchemy import func
from sqlalchemy.orm import Session

from admin.database import get_db
from admin.models.order import Order
from admin.models.user import User
from admin.models.sector import Sector
from admin.schemas.dashboard import DashboardStats, RecentOrdersResponse
from admin.dependencies import get_current_user

router = APIRouter()


@router.get("/stats", response_model=DashboardStats)
def get_stats(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    total_orders = db.query(Order).count()
    total_revenue = db.query(func.coalesce(func.sum(Order.total_amount), 0)).scalar()

    today_start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    orders_today = db.query(Order).filter(Order.created_at >= today_start).count()
    revenue_today = db.query(func.coalesce(func.sum(Order.total_amount), 0)).filter(Order.created_at >= today_start).scalar()

    status_rows = db.query(Order.status, func.count(Order.id)).group_by(Order.status).all()
    orders_by_status = {row[0]: row[1] for row in status_rows}

    sector_rows = db.query(Order.sector, func.count(Order.id)).group_by(Order.sector).all()
    orders_by_sector = {row[0]: row[1] for row in sector_rows}

    total_users = db.query(User).count()
    active_sectors = db.query(Sector).filter(Sector.is_active == True).count()

    return DashboardStats(
        total_orders=total_orders,
        total_revenue=float(total_revenue),
        orders_today=orders_today,
        revenue_today=float(revenue_today),
        orders_by_status=orders_by_status,
        orders_by_sector=orders_by_sector,
        total_users=total_users,
        active_sectors=active_sectors,
    )


@router.get("/recent-orders", response_model=RecentOrdersResponse)
def get_recent_orders(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    orders = db.query(Order).order_by(Order.created_at.desc()).limit(10).all()
    return {"orders": orders}
