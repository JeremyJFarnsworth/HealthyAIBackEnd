"""FastAPI application entrypoint.

Run with: uvicorn healthy.api.app:app --reload
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from healthy.api.routes import admin, auth, chat
from healthy.db.base import Base
from healthy.db.session import engine


@asynccontextmanager
async def lifespan(app: FastAPI):
    Base.metadata.create_all(bind=engine)
    yield


app = FastAPI(title="Healthy API", lifespan=lifespan)

app.include_router(auth.router)
app.include_router(admin.router)
app.include_router(chat.router)
