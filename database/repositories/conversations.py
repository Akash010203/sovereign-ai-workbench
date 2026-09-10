"""
database/repositories/conversations.py — Repository for conversations and messages.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime
from typing import Optional

from database.db import Database


class ConversationRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def create(self, title: str = "", model_name: str = "") -> str:
        conv_id = str(uuid.uuid4())
        self.db.insert("conversations", {
            "id": conv_id, "title": title, "model_name": model_name,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        })
        return conv_id

    def add_message(
        self, conversation_id: str, role: str, content: str, metadata: dict = None
    ) -> int:
        return self.db.insert("messages", {
            "conversation_id": conversation_id,
            "role": role,
            "content": content,
            "created_at": datetime.utcnow().isoformat(),
            "metadata": json.dumps(metadata or {}),
        })

    def get_messages(self, conversation_id: str) -> list[dict]:
        return self.db.fetch_all(
            "SELECT * FROM messages WHERE conversation_id=? ORDER BY id",
            (conversation_id,)
        )

    def list_conversations(self, limit: int = 50) -> list[dict]:
        return self.db.fetch_all(
            "SELECT * FROM conversations ORDER BY updated_at DESC LIMIT ?", (limit,)
        )

    def delete_conversation(self, conversation_id: str) -> None:
        self.db.execute("DELETE FROM messages WHERE conversation_id=?", (conversation_id,))
        self.db.execute("DELETE FROM conversations WHERE id=?", (conversation_id,))


class TaskRepository:
    def __init__(self, db: Database) -> None:
        self.db = db

    def save_task(self, task_state) -> None:
        """Upsert a TaskState to the database."""
        from agents.state import TaskState
        existing = self.db.fetch_one("SELECT id FROM tasks WHERE id=?", (task_state.task_id,))
        data = {
            "id":           task_state.task_id,
            "user_input":   task_state.user_input,
            "status":       task_state.status.value,
            "model_name":   task_state.model_name,
            "category":     task_state.category,
            "final_answer": task_state.final_answer,
            "error":        task_state.error,
            "updated_at":   datetime.utcnow().isoformat(),
        }
        if existing:
            sets = ", ".join(f"{k}=?" for k in data if k != "id")
            vals = [v for k, v in data.items() if k != "id"] + [task_state.task_id]
            self.db.execute(f"UPDATE tasks SET {sets} WHERE id=?", tuple(vals))
        else:
            data["created_at"] = datetime.utcnow().isoformat()
            self.db.insert("tasks", data)

        # Save steps
        for step in task_state.plan:
            self.db.insert("task_steps", {
                "task_id":     task_state.task_id,
                "step_number": step.step_number,
                "description": step.description,
                "tool_name":   step.tool_name,
                "tool_args":   json.dumps(step.tool_args),
                "result":      json.dumps(step.result)[:2000] if step.result else None,
                "success":     1 if step.success else 0,
                "error":       step.error,
                "created_at":  step.timestamp,
            })

    def get_task(self, task_id: str) -> Optional[dict]:
        return self.db.fetch_one("SELECT * FROM tasks WHERE id=?", (task_id,))

    def list_tasks(self, limit: int = 20) -> list[dict]:
        return self.db.fetch_all(
            "SELECT * FROM tasks ORDER BY created_at DESC LIMIT ?", (limit,)
        )
