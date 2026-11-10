"""Chat-history endpoints: submit a case, list and view past conversations.

``diagnose_from_text`` stays a pure, stateless function used identically by
the CLI and this route -- persistence happens only here, not inside
``healthy.llm`` (Requirement 9). The ``user_id`` filter on list/detail is the
server-side Requirement-11 enforcement point: a caller only ever sees their
own conversations, regardless of role.
"""

from __future__ import annotations

import json

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from healthy.api.schemas import ChatRequest, ConversationDetailOut, ConversationOut
from healthy.auth.deps import get_current_user
from healthy.db.models import ChatMessage, Conversation, User
from healthy.db.session import get_db
from healthy.llm.reasoning import diagnose_from_text

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ConversationDetailOut, status_code=status.HTTP_201_CREATED)
def submit_case(
    payload: ChatRequest,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationDetailOut:
    diagnosis = diagnose_from_text(payload.case_text)

    conversation = Conversation(user_id=current_user.id, case_text=payload.case_text)
    conversation.messages = [
        ChatMessage(role="user", content=payload.case_text),
        ChatMessage(role="assistant", content=json.dumps(diagnosis)),
    ]
    db.add(conversation)
    db.commit()
    db.refresh(conversation)

    return ConversationDetailOut(
        id=conversation.id,
        case_text=conversation.case_text,
        created_at=conversation.created_at,
        diagnosis=diagnosis,
    )


def _top_result(conversation: Conversation) -> dict | None:
    """Highest-ranked diagnosis from the stored assistant message, if any."""
    for message in conversation.messages:
        if message.role == "assistant":
            try:
                results = json.loads(message.content).get("results") or []
            except (ValueError, AttributeError):
                return None
            return results[0] if results else None
    return None


@router.get("", response_model=list[ConversationOut])
def list_conversations(
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user)
) -> list[ConversationOut]:
    conversations = (
        db.query(Conversation)
        .filter(Conversation.user_id == current_user.id)
        .order_by(Conversation.created_at.desc(), Conversation.id.desc())
        .all()
    )
    out = []
    for c in conversations:
        top = _top_result(c)
        out.append(
            ConversationOut(
                id=c.id,
                case_text=c.case_text,
                created_at=c.created_at,
                top_diagnosis=top["diagnosis"] if top else None,
                top_score=top["score"] if top else None,
            )
        )
    return out


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
def get_conversation(
    conversation_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ConversationDetailOut:
    conversation = (
        db.query(Conversation)
        .filter(
            Conversation.id == conversation_id,
            Conversation.user_id == current_user.id,
        )
        .first()
    )
    if conversation is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Conversation not found")

    assistant_message = next(m for m in conversation.messages if m.role == "assistant")
    return ConversationDetailOut(
        id=conversation.id,
        case_text=conversation.case_text,
        created_at=conversation.created_at,
        diagnosis=json.loads(assistant_message.content),
    )
