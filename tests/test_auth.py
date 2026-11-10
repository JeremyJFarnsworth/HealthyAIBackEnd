"""Unit tests for password hashing, JWT tokens, and the auth dependencies."""

import time

import pytest
from fastapi import HTTPException
from jose import jwt

from healthy.auth.deps import get_current_user, require_admin
from healthy.auth.security import (
    create_access_token,
    decode_access_token,
    hash_password,
    verify_password,
)
from healthy.config import settings
from healthy.db.models import User


def test_hash_password_round_trip():
    password_hash = hash_password("correct horse battery staple")

    assert password_hash != "correct horse battery staple"
    assert verify_password("correct horse battery staple", password_hash)
    assert not verify_password("wrong password", password_hash)


def test_create_and_decode_access_token_round_trip():
    token = create_access_token({"sub": "1", "role": "user"})

    payload = decode_access_token(token)

    assert payload["sub"] == "1"
    assert payload["role"] == "user"


def test_decode_access_token_rejects_tampered_token():
    token = create_access_token({"sub": "1", "role": "user"})

    assert decode_access_token(token + "tampered") is None


def test_decode_access_token_rejects_expired_token():
    expired_payload = {"sub": "1", "role": "user", "exp": time.time() - 10}
    expired_token = jwt.encode(
        expired_payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm
    )

    assert decode_access_token(expired_token) is None


def test_get_current_user_returns_user_for_valid_token(db_session, regular_user):
    token = create_access_token({"sub": str(regular_user.id), "role": regular_user.role})

    user = get_current_user(token=token, db=db_session)

    assert user.id == regular_user.id


def test_get_current_user_rejects_invalid_token(db_session):
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(token="not-a-real-token", db=db_session)

    assert exc_info.value.status_code == 401


def test_require_admin_allows_admin_role():
    admin = User(id=1, username="admin", password_hash="x", role="admin")

    assert require_admin(current_user=admin) is admin


def test_require_admin_rejects_non_admin_role():
    user = User(id=1, username="alice", password_hash="x", role="user")

    with pytest.raises(HTTPException) as exc_info:
        require_admin(current_user=user)

    assert exc_info.value.status_code == 403
