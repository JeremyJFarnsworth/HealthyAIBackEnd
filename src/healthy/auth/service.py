"""User-account operations shared by the auth and admin routes."""

from __future__ import annotations

from sqlalchemy.orm import Session

from healthy.auth.security import hash_password, verify_password
from healthy.db.models import User


def get_user_by_username(db: Session, username: str) -> User | None:
    return db.query(User).filter(User.username == username).first()


def get_user_by_id(db: Session, user_id: int) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def create_user(db: Session, username: str, password: str) -> User:
    user = User(username=username, password_hash=hash_password(password), role="user")
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def authenticate_user(db: Session, username: str, password: str) -> User | None:
    user = get_user_by_username(db, username)
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user
