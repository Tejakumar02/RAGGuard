import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from ragguard import api  # noqa: E402


@pytest.fixture
def client(monkeypatch, pipeline):
    monkeypatch.setitem(api.app.dependency_overrides, api.get_pipeline, lambda: pipeline)
    yield TestClient(api.app)
    api.app.dependency_overrides.clear()


def test_health(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert body["chunks"] > 0


def test_ask_returns_cited_answer(client):
    response = client.post("/ask", json={"question": "What torque are drain plugs tightened to?"})
    assert response.status_code == 200
    body = response.json()
    assert "30 Nm" in body["answer"]
    assert body["citations"]
    assert body["sources"][0]["source"] == "torque_and_fastening.md"


def test_ask_validates_input(client):
    assert client.post("/ask", json={"question": "hi"}).status_code == 422
    assert client.post("/ask", json={"question": "valid question", "top_k": 50}).status_code == 422
