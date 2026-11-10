"""Admin-only endpoints: list users, toggle role, delete a non-admin account.

Deleting a user cascades to their chat history via the ``User.conversations``
relationship's ``cascade="all, delete-orphan"`` (see ``healthy.db.models``) --
no manual multi-table delete is needed.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from healthy.auth.deps import require_admin
from healthy.auth.schemas import RoleUpdate, UserOut
from healthy.auth.service import get_user_by_id
from healthy.db.models import User
from healthy.db.session import get_db

router = APIRouter(prefix="/admin", tags=["admin"], dependencies=[Depends(require_admin)])


@router.get("/users", response_model=list[UserOut])
def list_users(db: Session = Depends(get_db)) -> list[User]:
    return db.query(User).all()


@router.patch("/users/{user_id}/role", response_model=UserOut)
def update_user_role(
    user_id: int, payload: RoleUpdate, db: Session = Depends(get_db)
) -> User:
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")

    user.role = payload.role
    db.commit()
    db.refresh(user)
    return user


@router.delete("/users/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_user(user_id: int, db: Session = Depends(get_db)) -> None:
    user = get_user_by_id(db, user_id)
    if user is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    if user.role == "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Cannot delete an admin account"
        )

    db.delete(user)
    db.commit()
