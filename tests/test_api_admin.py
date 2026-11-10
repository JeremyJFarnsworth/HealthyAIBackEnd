"""Integration tests for admin-only role-toggle and delete endpoints."""

from healthy.db.models import Conversation, User


def test_non_admin_cannot_access_admin_endpoints(client, auth_headers):
    response = client.get("/admin/users", headers=auth_headers)

    assert response.status_code == 403


def test_admin_can_list_users(client, admin_auth_headers, regular_user):
    response = client.get("/admin/users", headers=admin_auth_headers)

    assert response.status_code == 200
    usernames = [u["username"] for u in response.json()]
    assert "alice" in usernames


def test_admin_can_toggle_user_role(client, admin_auth_headers, regular_user):
    response = client.patch(
        f"/admin/users/{regular_user.id}/role",
        json={"role": "admin"},
        headers=admin_auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["role"] == "admin"

    response = client.patch(
        f"/admin/users/{regular_user.id}/role",
        json={"role": "user"},
        headers=admin_auth_headers,
    )

    assert response.status_code == 200
    assert response.json()["role"] == "user"


def test_role_toggle_rejects_non_admin_caller(client, auth_headers, regular_user):
    response = client.patch(
        f"/admin/users/{regular_user.id}/role",
        json={"role": "admin"},
        headers=auth_headers,
    )

    assert response.status_code == 403


def test_role_toggle_returns_404_for_missing_user(client, admin_auth_headers):
    response = client.patch(
        "/admin/users/999999/role", json={"role": "admin"}, headers=admin_auth_headers
    )

    assert response.status_code == 404


def test_admin_can_delete_non_admin_user_and_cascades_chat_history(
    client, admin_auth_headers, regular_user, db_session
):
    db_session.add(Conversation(user_id=regular_user.id, case_text="fever"))
    db_session.commit()

    response = client.delete(f"/admin/users/{regular_user.id}", headers=admin_auth_headers)

    assert response.status_code == 204
    assert db_session.get(User, regular_user.id) is None
    assert db_session.query(Conversation).filter_by(user_id=regular_user.id).count() == 0


def test_cannot_delete_an_admin_account(client, admin_auth_headers, admin_user):
    response = client.delete(f"/admin/users/{admin_user.id}", headers=admin_auth_headers)

    assert response.status_code == 403


def test_delete_rejects_non_admin_caller(client, auth_headers, regular_user):
    response = client.delete(f"/admin/users/{regular_user.id}", headers=auth_headers)

    assert response.status_code == 403
