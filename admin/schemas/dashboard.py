from pydantic import BaseModel
from admin.schemas.order import OrderResponse


class DashboardStats(BaseModel):
    total_orders: int
    total_revenue: float
    orders_today: int
    revenue_today: float
    orders_by_status: dict[str, int]
    orders_by_sector: dict[str, int]
    total_users: int
    active_sectors: int


class RecentOrdersResponse(BaseModel):
    orders: list[OrderResponse]
