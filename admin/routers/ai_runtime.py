from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from admin.database import get_db
from admin.dependencies import get_current_user
from admin.models.audit_log import AuditLog
from admin.models.user import User
from admin.services.openrouter import request_service_order_readiness_async
from admin.native_advisory_execution import run_native, connected

router = APIRouter()


class ServiceOrderReadinessRequest(BaseModel):
    order_summary: str = Field(min_length=10, max_length=1000)


@router.post("/service-order-readiness")
async def service_order_readiness(
    request: ServiceOrderReadinessRequest,
    transport: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        evidence = await run_native(transport, current_user.id, lambda: request_service_order_readiness_async(request.order_summary.strip()), timeout=45)
    except HTTPException as error:
        if error.status_code in (503, 504):
            return {"available": False, "message": "The AI provider is unavailable. No report was generated or saved. Please try again later."}
        raise
    except Exception:
        return {"available": False, "message": "The AI provider is unavailable. No report was generated or saved. Please try again later."}
    try:
        await connected(transport)
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
        await connected(transport)
        db.commit()
        db.refresh(audit)
        return {"analysisId": audit.id, **evidence}
    except HTTPException:
        db.rollback()
        raise
    except Exception as error:
        db.rollback()
        raise HTTPException(status_code=502, detail="AI provider request failed") from error

@router.get("/service-order-readiness")
async def service_order_history(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows=db.query(AuditLog).filter(AuditLog.user_id==current_user.id, AuditLog.action=="AI_SERVICE_ORDER_READINESS", AuditLog.entity_type=="ai_analysis").order_by(AuditLog.created_at.desc()).limit(50).all()
    return {"reports":[{"id":str(row.id),"prompt":row.details.get("orderSummary",""),"content":row.details.get("result",""),"model":row.details.get("providerReceipt",{}).get("model",""),"createdAt":row.created_at.isoformat()} for row in rows]}
