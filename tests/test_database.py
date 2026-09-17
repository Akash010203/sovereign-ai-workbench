"""
tests/test_database.py — Unit tests for the database module.

Tests: Database connection, schema creation, and repositories.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from database.db import Database
from database.repositories.conversations import ConversationRepository, TaskRepository


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def db(tmp_path):
    """Create a fresh in-memory database for each test."""
    db_path = tmp_path / "test.db"
    return Database(str(db_path))


@pytest.fixture
def conv_repo(db):
    return ConversationRepository(db)


@pytest.fixture
def task_repo(db):
    return TaskRepository(db)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Database basics
# ─────────────────────────────────────────────────────────────────────────────

class TestDatabase:
    def test_connection(self, db):
        """Database should initialize and create tables."""
        assert db is not None

    def test_insert_and_fetch(self, db):
        """Insert a record and fetch it back."""
        db.insert("conversations", {
            "id": "test-conv-1",
            "title": "Test Conversation",
            "model_name": "test_model",
            "created_at": "2026-01-01T00:00:00",
            "updated_at": "2026-01-01T00:00:00",
        })
        result = db.fetch_one(
            "SELECT * FROM conversations WHERE id=?", ("test-conv-1",)
        )
        assert result is not None
        assert result["title"] == "Test Conversation"


# ─────────────────────────────────────────────────────────────────────────────
# 2. ConversationRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestConversationRepository:
    def test_create_conversation(self, conv_repo):
        conv_id = conv_repo.create(title="Test Chat", model_name="minilm")
        assert isinstance(conv_id, str)
        assert len(conv_id) > 0

    def test_add_and_get_messages(self, conv_repo):
        conv_id = conv_repo.create(title="Chat")
        conv_repo.add_message(conv_id, "user", "Hello!")
        conv_repo.add_message(conv_id, "assistant", "Hi there!")

        messages = conv_repo.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_list_conversations(self, conv_repo):
        conv_repo.create(title="Chat 1")
        conv_repo.create(title="Chat 2")

        convs = conv_repo.list_conversations()
        assert len(convs) >= 2

    def test_delete_conversation(self, conv_repo):
        conv_id = conv_repo.create(title="To Delete")
        conv_repo.add_message(conv_id, "user", "bye")
        conv_repo.delete_conversation(conv_id)

        messages = conv_repo.get_messages(conv_id)
        assert len(messages) == 0


# ─────────────────────────────────────────────────────────────────────────────
# 3. TaskRepository
# ─────────────────────────────────────────────────────────────────────────────

class TestTaskRepository:
    def test_save_and_get_task(self, task_repo):
        from agents.state import TaskState, TaskStatus, AgentStep

        state = TaskState(
            task_id="task-test-001",
            user_input="Calculate 2+2",
            category="calculation",
        )
        state.update_status(TaskStatus.DONE)
        state.final_answer = "4"
        state.plan = [
            AgentStep(step_number=1, description="calc", tool_name="calculator",
                      tool_args={"expression": "2+2"}, result="4", success=True),
        ]

        task_repo.save_task(state)

        saved = task_repo.get_task("task-test-001")
        assert saved is not None
        assert saved["status"] == "done"
        assert saved["final_answer"] == "4"

    def test_list_tasks(self, task_repo):
        from agents.state import TaskState, TaskStatus

        for i in range(3):
            state = TaskState(task_id=f"task-{i}", user_input=f"task {i}")
            state.update_status(TaskStatus.DONE)
            task_repo.save_task(state)

        tasks = task_repo.list_tasks(limit=10)
        assert len(tasks) >= 3

    def test_get_nonexistent_task(self, task_repo):
        result = task_repo.get_task("does-not-exist")
        assert result is None
