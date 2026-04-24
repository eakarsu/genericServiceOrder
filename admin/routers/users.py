import math
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session

from admin.database import get_db
from admin.models.user import User
from admin.models.role import Role
from admin.models.audit_log import AuditLog
from admin.schemas.user import UserResponse, UserListResponse, UserCreate, UserUpdate, ProfileUpdate
from admin.services.auth_service import hash_password
from admin.services.password_service import validate_password_strength
from admin.dependencies import get_current_user, require_role

router = APIRouter()


@router.get("", response_model=UserListResponse)
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    search: str | None = None,
    role_id: int | None = None,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    query = db.query(User)

    if search:
        like = f"%{search}%"
        query = query.filter(
            (User.name.ilike(like)) | (User.email.ilike(like))
        )
    if role_id:
        query = query.filter(User.role_id == role_id)

    total = query.count()
    users = query.order_by(User.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()

    user_responses = []
    for u in users:
        role = db.query(Role).filter(Role.id == u.role_id).first()
        user_responses.append(UserResponse(
            id=u.id, email=u.email, name=u.name, phone=u.phone,
            role_id=u.role_id, role_name=role.name if role else None,
            is_active=u.is_active, is_email_verified=u.is_email_verified,
            created_at=u.created_at, updated_at=u.updated_at,
        ))

    return {
        "users": user_responses,
        "total": total,
        "page": page,
        "page_size": page_size,
        "total_pages": math.ceil(total / page_size) if total > 0 else 1,
    }


@router.post("", response_model=UserResponse)
def create_user(
    req: UserCreate,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    is_valid, msg = validate_password_strength(req.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        name=req.name,
        phone=req.phone,
        role_id=req.role_id,
        is_email_verified=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    role = db.query(Role).filter(Role.id == user.role_id).first()
    db.add(AuditLog(user_id=current_user.id, action="create", entity_type="user", entity_id=user.id))
    db.commit()

    return UserResponse(
        id=user.id, email=user.email, name=user.name, phone=user.phone,
        role_id=user.role_id, role_name=role.name if role else None,
        is_active=user.is_active, is_email_verified=user.is_email_verified,
        created_at=user.created_at, updated_at=user.updated_at,
    )


@router.get("/{user_id}", response_model=UserResponse)
def get_user(
    user_id: int,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    role = db.query(Role).filter(Role.id == user.role_id).first()
    return UserResponse(
        id=user.id, email=user.email, name=user.name, phone=user.phone,
        role_id=user.role_id, role_name=role.name if role else None,
        is_active=user.is_active, is_email_verified=user.is_email_verified,
        created_at=user.created_at, updated_at=user.updated_at,
    )


@router.put("/{user_id}", response_model=UserResponse)
def update_user(
    user_id: int,
    req: UserUpdate,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(user, key, value)

    db.add(AuditLog(user_id=current_user.id, action="update", entity_type="user", entity_id=user_id, details=update_data))
    db.commit()
    db.refresh(user)

    role = db.query(Role).filter(Role.id == user.role_id).first()
    return UserResponse(
        id=user.id, email=user.email, name=user.name, phone=user.phone,
        role_id=user.role_id, role_name=role.name if role else None,
        is_active=user.is_active, is_email_verified=user.is_email_verified,
        created_at=user.created_at, updated_at=user.updated_at,
    )


@router.delete("/{user_id}")
def delete_user(
    user_id: int,
    current_user: User = Depends(require_role("admin")),
    db: Session = Depends(get_db),
):
    if user_id == current_user.id:
        raise HTTPException(status_code=400, detail="Cannot delete yourself")

    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    db.add(AuditLog(user_id=current_user.id, action="delete", entity_type="user", entity_id=user_id))
    db.delete(user)
    db.commit()
    return {"message": "User deleted"}


@router.get("/me/profile", response_model=UserResponse)
def get_profile(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    role = db.query(Role).filter(Role.id == current_user.role_id).first()
    return UserResponse(
        id=current_user.id, email=current_user.email, name=current_user.name,
        phone=current_user.phone, role_id=current_user.role_id,
        role_name=role.name if role else None, is_active=current_user.is_active,
        is_email_verified=current_user.is_email_verified,
        created_at=current_user.created_at, updated_at=current_user.updated_at,
    )


@router.put("/me/profile", response_model=UserResponse)
def update_profile(
    req: ProfileUpdate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    update_data = req.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        setattr(current_user, key, value)
    db.commit()
    db.refresh(current_user)

    role = db.query(Role).filter(Role.id == current_user.role_id).first()
    return UserResponse(
        id=current_user.id, email=current_user.email, name=current_user.name,
        phone=current_user.phone, role_id=current_user.role_id,
        role_name=role.name if role else None, is_active=current_user.is_active,
        is_email_verified=current_user.is_email_verified,
        created_at=current_user.created_at, updated_at=current_user.updated_at,
    )
