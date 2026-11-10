"""FastAPI dependencies enforcing authentication and role-based access (Requirement 11)."""

from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from healthy.auth.security import decode_access_token
from healthy.auth.service import get_user_by_id
from healthy.db.models import User
from healthy.db.session import get_db

_oauth2_scheme = OAuth2PasswordBearer(tokenUrl="auth/login")

_CREDENTIALS_ERROR = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)


def get_current_user(
    token: str = Depends(_oauth2_scheme), db: Session = Depends(get_db)
) -> User:
    payload = decode_access_token(token)
    if payload is None:
        raise _CREDENTIALS_ERROR

    user_id = payload.get("sub")
    if user_id is None:
        raise _CREDENTIALS_ERROR

    user = get_user_by_id(db, int(user_id))
    if user is None:
        raise _CREDENTIALS_ERROR
    return user


def require_admin(current_user: User = Depends(get_current_user)) -> User:
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin privileges required",
        )
    return current_user
