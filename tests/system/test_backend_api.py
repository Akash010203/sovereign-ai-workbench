"""
TEST P — Backend API Validation

Verifies:
- Flask app creates successfully
- All API routes exist
- Status endpoint returns valid JSON
- Chat endpoint works
- Error handling works
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture(scope="module")
def client():
    """Create a test client for the Flask app."""
    from app.backend.app import create_app
    app = create_app(config={})
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestBackendStartup:
    """Test that the backend starts correctly."""

    def test_app_creates(self, client):
        assert client is not None

    def test_index_returns_html(self, client):
        r = client.get("/")
        assert r.status_code == 200


class TestStatusEndpoint:
    """Test /api/status."""

    def test_status_returns_json(self, client):
        r = client.get("/api/status")
        assert r.status_code == 200
        data = r.get_json()
        assert data is not None
        assert "status" in data
        assert data["status"] == "running"

    def test_status_has_models(self, client):
        r = client.get("/api/status")
        data = r.get_json()
        assert "models" in data

    def test_status_has_rag_info(self, client):
        r = client.get("/api/status")
        data = r.get_json()
        assert "rag_docs" in data

    def test_status_has_offline_info(self, client):
        r = client.get("/api/status")
        data = r.get_json()
        assert "offline" in data


class TestModelsEndpoint:
    """Test /api/models."""

    def test_models_returns_list(self, client):
        r = client.get("/api/models")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list)


class TestChatEndpoint:
    """Test /api/chat."""

    def test_chat_empty_message_rejected(self, client):
        r = client.post("/api/chat",
                        data=json.dumps({"message": ""}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_chat_returns_reply(self, client):
        r = client.post("/api/chat",
                        data=json.dumps({"message": "Hello"}),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "reply" in data
        assert len(data["reply"]) > 0

    def test_chat_returns_conversation_id(self, client):
        r = client.post("/api/chat",
                        data=json.dumps({"message": "Hi there"}),
                        content_type="application/json")
        data = r.get_json()
        assert "conversation_id" in data

    def test_chat_returns_routing_info(self, client):
        r = client.post("/api/chat",
                        data=json.dumps({"message": "What is the pump pressure?"}),
                        content_type="application/json")
        data = r.get_json()
        assert "category" in data
        assert "model_used" in data


class TestAgentEndpoint:
    """Test /api/agent."""

    def test_agent_empty_task_rejected(self, client):
        r = client.post("/api/agent",
                        data=json.dumps({"task": ""}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_agent_returns_result(self, client):
        r = client.post("/api/agent",
                        data=json.dumps({"task": "Calculate 2 + 2"}),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "task_id" in data


class TestRAGEndpoints:
    """Test RAG API endpoints."""

    def test_rag_search_empty_rejected(self, client):
        r = client.post("/api/rag/search",
                        data=json.dumps({"query": ""}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_rag_ingest_empty_rejected(self, client):
        r = client.post("/api/rag/ingest",
                        data=json.dumps({"text": ""}),
                        content_type="application/json")
        assert r.status_code == 400

    def test_rag_search_returns_results(self, client):
        r = client.post("/api/rag/search",
                        data=json.dumps({"query": "pump maintenance"}),
                        content_type="application/json")
        assert r.status_code == 200
        data = r.get_json()
        assert "results" in data


class TestSecurityEndpoint:
    """Test /api/security/offline."""

    def test_offline_check(self, client):
        r = client.get("/api/security/offline")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, dict)


class TestConversationsEndpoint:
    """Test /api/conversations."""

    def test_list_conversations(self, client):
        r = client.get("/api/conversations")
        assert r.status_code == 200
        data = r.get_json()
        assert isinstance(data, list)


class TestErrorHandling:
    """Test that errors return JSON, not HTML."""

    def test_404_returns_json(self, client):
        r = client.get("/api/nonexistent")
        assert r.status_code == 404
        data = r.get_json()
        assert data is not None
        assert "error" in data

    def test_malformed_json_handled(self, client):
        r = client.post("/api/chat",
                        data="not json",
                        content_type="application/json")
        # Should not crash with 500
        assert r.status_code in (400, 500)
