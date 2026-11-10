"""Shared fixtures for the DB/auth/API test suite (Week 5).

Sets ``JWT_SECRET_KEY`` before any ``healthy`` import since
``healthy.config.Settings`` requires it with no default. Every test gets an
isolated in-memory SQLite database via ``db_session``/``client`` -- no test
touches a real Postgres instance or the local LLM server.
"""

import os

os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", "sqlite://")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from healthy.api.app import app
from healthy.auth.security import create_access_token, hash_password
from healthy.db.base import Base
from healthy.db.models import User
from healthy.db.session import get_db


@pytest.fixture()
def db_session():
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def _get_db_override():
        yield db_session

    app.dependency_overrides[get_db] = _get_db_override
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def _make_user(db_session, username: str, role: str) -> User:
    user = User(username=username, password_hash=hash_password("password123"), role=role)
    db_session.add(user)
    db_session.commit()
    db_session.refresh(user)
    return user


def _auth_header(user: User) -> dict:
    token = create_access_token({"sub": str(user.id), "role": user.role})
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def regular_user(db_session) -> User:
    return _make_user(db_session, "alice", "user")


@pytest.fixture()
def admin_user(db_session) -> User:
    return _make_user(db_session, "admin", "admin")


@pytest.fixture()
def auth_headers(regular_user) -> dict:
    return _auth_header(regular_user)


@pytest.fixture()
def admin_auth_headers(admin_user) -> dict:
    return _auth_header(admin_user)
