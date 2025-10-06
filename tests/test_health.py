from fastapi.testclient import TestClient

import app as search_app


def test_health_endpoint(monkeypatch):
    client = TestClient(search_app.app)

    monkeypatch.setattr(
        search_app,
        "es",
        type("FakeES", (), {"info": lambda self: {"version": {"number": "8.0.0"}}})(),
    )

    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "es_version" in payload
