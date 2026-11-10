"""Integration tests for the registration and login endpoints."""


def test_register_creates_user_with_user_role(client):
    response = client.post(
        "/auth/register", json={"username": "newuser", "password": "password123"}
    )

    assert response.status_code == 201
    body = response.json()
    assert body["username"] == "newuser"
    assert body["role"] == "user"
    assert "password" not in body
    assert "password_hash" not in body


def test_register_rejects_duplicate_username(client):
    client.post("/auth/register", json={"username": "dup", "password": "password123"})

    response = client.post("/auth/register", json={"username": "dup", "password": "password456"})

    assert response.status_code == 409


def test_login_succeeds_with_correct_credentials(client):
    client.post("/auth/register", json={"username": "loginuser", "password": "password123"})

    response = client.post(
        "/auth/login", data={"username": "loginuser", "password": "password123"}
    )

    assert response.status_code == 200
    body = response.json()
    assert body["token_type"] == "bearer"
    assert body["access_token"]


def test_login_rejects_wrong_password(client):
    client.post("/auth/register", json={"username": "loginuser2", "password": "password123"})

    response = client.post(
        "/auth/login", data={"username": "loginuser2", "password": "wrongpassword"}
    )

    assert response.status_code == 401


def test_login_rejects_unknown_user(client):
    response = client.post(
        "/auth/login", data={"username": "nobody", "password": "password123"}
    )

    assert response.status_code == 401
