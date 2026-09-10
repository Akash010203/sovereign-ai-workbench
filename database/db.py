"""
database/db.py — SQLite connection manager and schema initializer.

Uses Python's built-in sqlite3 — no ORM, no extra dependencies.
"""
from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data") / "sovereign_ai.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open a SQLite connection with row_factory for dict-like access.
    Creates the database file if it doesn't exist.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db(db_path: str | Path = DEFAULT_DB_PATH) -> None:
    """Create all tables if they don't exist by running schema.sql."""
    schema_path = Path(__file__).parent / "schema.sql"
    if not schema_path.exists():
        log.error("schema.sql not found at %s", schema_path)
        return
    schema_sql = schema_path.read_text(encoding="utf-8")
    conn = get_connection(db_path)
    conn.executescript(schema_sql)
    conn.commit()
    conn.close()
    log.info("Database initialized at %s", db_path)


class Database:
    """
    Thin wrapper around a SQLite connection.

    Provides execute(), fetch_one(), fetch_all(), and insert() helpers
    so the repositories don't need to write raw sqlite3 boilerplate.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        init_db(self._path)
        self._conn = get_connection(self._path)

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        cursor = self._conn.execute(sql, params)
        self._conn.commit()
        return cursor

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        row = self._conn.execute(sql, params).fetchone()
        return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def insert(self, table: str, data: dict) -> int:
        """Insert a row and return the new rowid."""
        cols   = ", ".join(data.keys())
        places = ", ".join("?" * len(data))
        sql    = f"INSERT INTO {table} ({cols}) VALUES ({places})"
        cursor = self.execute(sql, tuple(data.values()))
        return cursor.lastrowid

    def close(self) -> None:
        self._conn.close()
