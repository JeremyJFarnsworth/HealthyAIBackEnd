"""Unit tests for the ORM models: uniqueness constraints and cascade deletes."""

import pytest
from sqlalchemy.exc import IntegrityError

from healthy.db.models import ChatMessage, Conversation, User


def test_username_must_be_unique(db_session):
    db_session.add(User(username="bob", password_hash="hash1", role="user"))
    db_session.commit()

    db_session.add(User(username="bob", password_hash="hash2", role="user"))
    with pytest.raises(IntegrityError):
        db_session.commit()


def test_deleting_user_cascades_to_conversations_and_messages(db_session):
    user = User(username="carol", password_hash="hash", role="user")
    db_session.add(user)
    db_session.commit()

    conversation = Conversation(user_id=user.id, case_text="I have a fever")
    conversation.messages = [
        ChatMessage(role="user", content="I have a fever"),
        ChatMessage(role="assistant", content='{"results": []}'),
    ]
    db_session.add(conversation)
    db_session.commit()
    conversation_id = conversation.id

    db_session.delete(user)
    db_session.commit()

    assert db_session.get(Conversation, conversation_id) is None
    assert db_session.query(ChatMessage).count() == 0
