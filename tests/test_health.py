import pytest

pytest.importorskip("httpx")

from fastapi.testclient import TestClient

import app as search_app


class FakeES:
    """Test double that mimics the subset of Elasticsearch client behavior used in health."""

    def info(self):
        return {"version": {"number": "8.0.0"}}


def test_health_endpoint(monkeypatch):
    monkeypatch.setattr(search_app, "es", FakeES())

    client = TestClient(search_app.app)
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["ok"] is True
    assert "es_version" in payload
