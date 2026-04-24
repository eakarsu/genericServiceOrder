import secrets
from datetime import datetime, timedelta, timezone
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from admin.database import get_db
from admin.models.user import User
from admin.models.role import Role
from admin.models.refresh_token import RefreshToken
from admin.schemas.auth import (
    RegisterRequest, LoginRequest, TokenResponse, RefreshRequest,
    ForgotPasswordRequest, ResetPasswordRequest, ChangePasswordRequest, MessageResponse,
)
from admin.services.auth_service import (
    hash_password, verify_password, create_access_token,
    create_refresh_token, decode_token, get_refresh_token_expiry,
)
from admin.services.password_service import validate_password_strength
from admin.services.email_service import send_verification_email, send_password_reset_email
from admin.dependencies import get_current_user

router = APIRouter()


@router.post("/register", response_model=MessageResponse)
def register(req: RegisterRequest, db: Session = Depends(get_db)):
    is_valid, msg = validate_password_strength(req.password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    if db.query(User).filter(User.email == req.email).first():
        raise HTTPException(status_code=400, detail="Email already registered")

    viewer_role = db.query(Role).filter(Role.name == "viewer").first()
    if not viewer_role:
        raise HTTPException(status_code=500, detail="Default role not found. Run seed first.")

    verification_token = secrets.token_urlsafe(32)
    user = User(
        email=req.email,
        password_hash=hash_password(req.password),
        name=req.name,
        phone=req.phone,
        role_id=viewer_role.id,
        email_verification_token=verification_token,
    )
    db.add(user)
    db.commit()

    send_verification_email(req.email, verification_token)
    return {"message": "Registration successful. Please verify your email."}


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if not user or not verify_password(req.password, user.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    access_token = create_access_token({"sub": str(user.id)})
    refresh_token_str = create_refresh_token({"sub": str(user.id)})

    db_refresh = RefreshToken(
        user_id=user.id,
        token=refresh_token_str,
        expires_at=get_refresh_token_expiry(),
    )
    db.add(db_refresh)
    db.commit()

    return {"access_token": access_token, "refresh_token": refresh_token_str}


@router.post("/logout", response_model=MessageResponse)
def logout(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    db.query(RefreshToken).filter(
        RefreshToken.user_id == current_user.id,
        RefreshToken.is_revoked == False,
    ).update({"is_revoked": True})
    db.commit()
    return {"message": "Logged out successfully"}


@router.post("/refresh", response_model=TokenResponse)
def refresh(req: RefreshRequest, db: Session = Depends(get_db)):
    payload = decode_token(req.refresh_token)
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid refresh token")

    db_token = db.query(RefreshToken).filter(
        RefreshToken.token == req.refresh_token,
        RefreshToken.is_revoked == False,
    ).first()
    if not db_token or db_token.expires_at < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Refresh token expired or revoked")

    # Revoke old token
    db_token.is_revoked = True

    user_id = payload["sub"]
    access_token = create_access_token({"sub": user_id})
    new_refresh = create_refresh_token({"sub": user_id})

    db.add(RefreshToken(
        user_id=int(user_id),
        token=new_refresh,
        expires_at=get_refresh_token_expiry(),
    ))
    db.commit()

    return {"access_token": access_token, "refresh_token": new_refresh}


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(req: ForgotPasswordRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email == req.email).first()
    if user:
        token = secrets.token_urlsafe(32)
        user.password_reset_token = token
        user.password_reset_expires = datetime.now(timezone.utc) + timedelta(hours=1)
        db.commit()
        send_password_reset_email(req.email, token)
    # Always return success to prevent email enumeration
    return {"message": "If the email exists, a password reset link has been sent."}


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(req: ResetPasswordRequest, db: Session = Depends(get_db)):
    is_valid, msg = validate_password_strength(req.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    user = db.query(User).filter(User.password_reset_token == req.token).first()
    if not user or not user.password_reset_expires or user.password_reset_expires < datetime.now(timezone.utc):
        raise HTTPException(status_code=400, detail="Invalid or expired reset token")

    user.password_hash = hash_password(req.new_password)
    user.password_reset_token = None
    user.password_reset_expires = None
    db.commit()
    return {"message": "Password reset successful"}


@router.post("/change-password", response_model=MessageResponse)
def change_password(
    req: ChangePasswordRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not verify_password(req.current_password, current_user.password_hash):
        raise HTTPException(status_code=400, detail="Current password is incorrect")

    is_valid, msg = validate_password_strength(req.new_password)
    if not is_valid:
        raise HTTPException(status_code=400, detail=msg)

    current_user.password_hash = hash_password(req.new_password)
    db.commit()
    return {"message": "Password changed successfully"}


@router.get("/verify-email/{token}", response_model=MessageResponse)
def verify_email(token: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.email_verification_token == token).first()
    if not user:
        raise HTTPException(status_code=400, detail="Invalid verification token")

    user.is_email_verified = True
    user.email_verification_token = None
    db.commit()
    return {"message": "Email verified successfully"}


@router.get("/me")
def get_me(current_user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    role = db.query(Role).filter(Role.id == current_user.role_id).first()
    return {
        "id": current_user.id,
        "email": current_user.email,
        "name": current_user.name,
        "phone": current_user.phone,
        "role_id": current_user.role_id,
        "role_name": role.name if role else None,
        "is_active": current_user.is_active,
        "is_email_verified": current_user.is_email_verified,
        "created_at": current_user.created_at,
        "updated_at": current_user.updated_at,
    }
