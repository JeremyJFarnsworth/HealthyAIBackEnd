"""Integration tests for the chat-history endpoints.

``diagnose_from_text`` is patched at its use-site in ``healthy.api.routes.chat``
(same convention as ``test_llm.py``'s patching of ``chat_completion_json``) so
these tests never load the model/vocab or call the local LLM server.
"""

from unittest.mock import patch

_FAKE_DIAGNOSIS = {
    "results": [
        {
            "diagnosis": "Psoriasis",
            "score": 0.7,
            "description": "A chronic skin condition.",
            "common_symptoms": ["skin rash", "itching"],
            "treatment_plan": "Topical corticosteroids.",
            "suggested_tests": ["Skin biopsy"],
        }
    ]
}


def test_submit_case_persists_and_returns_diagnosis(client, auth_headers):
    with patch(
        "healthy.api.routes.chat.diagnose_from_text", return_value=_FAKE_DIAGNOSIS
    ) as mock_diagnose:
        response = client.post(
            "/chat", json={"case_text": "I have a skin rash"}, headers=auth_headers
        )

    mock_diagnose.assert_called_once_with("I have a skin rash")
    assert response.status_code == 201
    body = response.json()
    assert body["case_text"] == "I have a skin rash"
    assert body["diagnosis"] == _FAKE_DIAGNOSIS


def test_submit_case_requires_authentication(client):
    response = client.post("/chat", json={"case_text": "I have a skin rash"})

    assert response.status_code == 401


def test_list_conversations_only_returns_callers_own(client, auth_headers, admin_auth_headers):
    with patch("healthy.api.routes.chat.diagnose_from_text", return_value=_FAKE_DIAGNOSIS):
        client.post("/chat", json={"case_text": "alice's case"}, headers=auth_headers)
        client.post("/chat", json={"case_text": "admin's case"}, headers=admin_auth_headers)

    response = client.get("/chat", headers=auth_headers)

    assert response.status_code == 200
    case_texts = [c["case_text"] for c in response.json()]
    assert case_texts == ["alice's case"]


def test_get_conversation_detail_for_owner(client, auth_headers):
    with patch("healthy.api.routes.chat.diagnose_from_text", return_value=_FAKE_DIAGNOSIS):
        created = client.post(
            "/chat", json={"case_text": "I have a skin rash"}, headers=auth_headers
        ).json()

    response = client.get(f"/chat/{created['id']}", headers=auth_headers)

    assert response.status_code == 200
    assert response.json()["diagnosis"] == _FAKE_DIAGNOSIS


def test_get_conversation_detail_404s_for_non_owner(client, auth_headers, admin_auth_headers):
    with patch("healthy.api.routes.chat.diagnose_from_text", return_value=_FAKE_DIAGNOSIS):
        created = client.post(
            "/chat", json={"case_text": "I have a skin rash"}, headers=auth_headers
        ).json()

    response = client.get(f"/chat/{created['id']}", headers=admin_auth_headers)

    assert response.status_code == 404


def test_get_conversation_detail_requires_authentication(client):
    response = client.get("/chat/1")

    assert response.status_code == 401


def test_list_conversations_includes_top_diagnosis_and_score(client, auth_headers):
    with patch("healthy.api.routes.chat.diagnose_from_text", return_value=_FAKE_DIAGNOSIS):
        client.post("/chat", json={"case_text": "I have a skin rash"}, headers=auth_headers)

    item = client.get("/chat", headers=auth_headers).json()[0]

    assert item["top_diagnosis"] == "Psoriasis"
    assert item["top_score"] == 0.7
