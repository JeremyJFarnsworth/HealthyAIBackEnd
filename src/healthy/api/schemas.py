"""Request/response contracts for the chat-history endpoints."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel

from healthy.llm.schemas import DiagnosisResponse


class ChatRequest(BaseModel):
    case_text: str


class ConversationOut(BaseModel):
    id: int
    case_text: str
    created_at: datetime
    top_diagnosis: str | None = None
    top_score: float | None = None

    model_config = {"from_attributes": True}


class ConversationDetailOut(BaseModel):
    id: int
    case_text: str
    created_at: datetime
    diagnosis: DiagnosisResponse
