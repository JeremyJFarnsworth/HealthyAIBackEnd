"""Shared runtime configuration for the persistence and auth layers (Week 5).

Read from environment variables, same convention as ``healthy.llm.client``'s
``LLM_BASE_URL``/``LLM_MODEL`` -- no secrets are hardcoded, and ``JWT_SECRET_KEY``
has no safe default so a deployment can't silently run with a known key.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "sqlite:///./healthy.db"
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60


settings = Settings()
