"""Pydantic request/response contracts for auth and user data."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class UserCreate(BaseModel):
    username: str
    password: str


class UserOut(BaseModel):
    id: int
    username: str
    role: Literal["user", "admin"]

    model_config = {"from_attributes": True}


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class RoleUpdate(BaseModel):
    role: Literal["user", "admin"]
