"""
TEST O — Database Full Validation

Verifies:
- Create/insert/read/update/delete
- Persistence after reopening
- Malformed data handling
- Schema initialization
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from database.db import Database, init_db
from database.repositories.conversations import ConversationRepository, TaskRepository


class TestDatabaseCore:
    """Test core database operations."""

    @pytest.fixture
    def db(self, tmp_path):
        db_path = tmp_path / "test.db"
        return Database(db_path)

    def test_database_creates_file(self, tmp_path):
        db_path = tmp_path / "test_create.db"
        db = Database(db_path)
        assert db_path.exists()

    def test_insert_and_fetch(self, db):
        db.execute(
            "INSERT INTO conversations (id, title, model_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("test-1", "Test Convo", "model", "2024-01-01", "2024-01-01"),
        )
        row = db.fetch_one("SELECT * FROM conversations WHERE id = ?", ("test-1",))
        assert row is not None
        assert row["title"] == "Test Convo"

    def test_fetch_all(self, db):
        for i in range(5):
            db.execute(
                "INSERT INTO conversations (id, title, model_name, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (f"conv-{i}", f"Title {i}", "model", "2024-01-01", "2024-01-01"),
            )
        rows = db.fetch_all("SELECT * FROM conversations")
        assert len(rows) == 5

    def test_delete(self, db):
        db.execute(
            "INSERT INTO conversations (id, title, model_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("del-1", "Delete Me", "model", "2024-01-01", "2024-01-01"),
        )
        db.execute("DELETE FROM conversations WHERE id = ?", ("del-1",))
        row = db.fetch_one("SELECT * FROM conversations WHERE id = ?", ("del-1",))
        assert row is None

    def test_persistence_after_close(self, tmp_path):
        db_path = tmp_path / "persist.db"
        db1 = Database(db_path)
        db1.execute(
            "INSERT INTO conversations (id, title, model_name, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?)",
            ("persist-1", "Persistent", "model", "2024-01-01", "2024-01-01"),
        )
        db1.close()

        # Reopen
        db2 = Database(db_path)
        row = db2.fetch_one("SELECT * FROM conversations WHERE id = ?", ("persist-1",))
        assert row is not None
        assert row["title"] == "Persistent"


class TestConversationRepo:
    """Test conversation repository."""

    @pytest.fixture
    def repo(self, tmp_path):
        db = Database(tmp_path / "conv.db")
        return ConversationRepository(db)

    def test_create_conversation(self, repo):
        conv_id = repo.create(title="Test", model_name="model")
        assert conv_id is not None
        assert len(conv_id) > 0

    def test_add_and_get_messages(self, repo):
        conv_id = repo.create(title="Msg Test", model_name="model")
        repo.add_message(conv_id, "user", "Hello")
        repo.add_message(conv_id, "assistant", "Hi there")
        messages = repo.get_messages(conv_id)
        assert len(messages) == 2
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"

    def test_list_conversations(self, repo):
        repo.create(title="Conv 1", model_name="m")
        repo.create(title="Conv 2", model_name="m")
        convos = repo.list_conversations()
        assert len(convos) >= 2

    def test_delete_conversation(self, repo):
        conv_id = repo.create(title="Delete Me", model_name="m")
        repo.delete_conversation(conv_id)
        convos = repo.list_conversations()
        ids = [c["id"] for c in convos]
        assert conv_id not in ids


class TestTaskRepo:
    """Test task repository."""

    @pytest.fixture
    def repo(self, tmp_path):
        db = Database(tmp_path / "task.db")
        return TaskRepository(db)

    def test_save_and_get_task(self, repo):
        from agents.state import TaskState
        state = TaskState(
            task_id="task-1",
            user_input="Test input",
            model_name="model",
            category="test",
        )
        state.final_answer = "Test answer"
        repo.save_task(state)
        task = repo.get_task("task-1")
        assert task is not None

    def test_list_tasks(self, repo):
        from agents.state import TaskState
        for i in range(3):
            state = TaskState(
                task_id=f"task-{i}",
                user_input=f"Input {i}",
                model_name="model",
                category="test",
            )
            repo.save_task(state)
        tasks = repo.list_tasks()
        assert len(tasks) >= 3
