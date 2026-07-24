from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from admin.database import get_db
from admin.dependencies import get_current_user
from admin.models.audit_log import AuditLog
from admin.models.user import User
from admin.services.openrouter import request_service_order_readiness

router = APIRouter()


class ServiceOrderReadinessRequest(BaseModel):
    order_summary: str = Field(min_length=10, max_length=1000)


@router.post("/service-order-readiness")
def service_order_readiness(
    request: ServiceOrderReadinessRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        evidence = request_service_order_readiness(request.order_summary.strip())
        audit = AuditLog(
            user_id=current_user.id,
            action="AI_SERVICE_ORDER_READINESS",
            entity_type="ai_analysis",
            details={
                "orderSummary": request.order_summary.strip(),
                "result": evidence["result"],
                "providerReceipt": evidence["providerReceipt"],
            },
        )
        db.add(audit)
        db.commit()
        db.refresh(audit)
        return {"analysisId": audit.id, **evidence}
    except RuntimeError as error:
        db.rollback()
        raise HTTPException(status_code=502, detail="AI provider request failed") from error
