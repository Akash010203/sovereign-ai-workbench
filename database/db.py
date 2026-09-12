"""
database/db.py — SQLite connection manager and schema initializer.

Uses Python's built-in sqlite3 — no ORM, no extra dependencies.
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
from pathlib import Path
from typing import Any, Optional

log = logging.getLogger(__name__)

DEFAULT_DB_PATH = Path("data") / "sovereign_ai.db"


def get_connection(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """
    Open a SQLite connection with row_factory for dict-like access.
    Creates the database file if it doesn't exist.
    check_same_thread=False allows worker threads in Flask to query cleanly.
    """
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path), check_same_thread=False, timeout=30.0)
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
    Thread-safe wrapper around a SQLite connection.

    Provides execute(), fetch_one(), fetch_all(), and insert() helpers.
    Uses threading.local() so each Flask worker thread safely manages its own
    connection without cross-thread SQLite exceptions.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self._path = Path(db_path)
        init_db(self._path)
        self._local = threading.local()
        self._lock = threading.Lock()

    @property
    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = get_connection(self._path)
            self._local.conn = conn
        return conn

    def execute(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        with self._lock:
            cursor = self._conn.execute(sql, params)
            self._conn.commit()
            return cursor

    def fetch_one(self, sql: str, params: tuple = ()) -> Optional[dict]:
        with self._lock:
            row = self._conn.execute(sql, params).fetchone()
            return dict(row) if row else None

    def fetch_all(self, sql: str, params: tuple = ()) -> list[dict]:
        with self._lock:
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
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            except Exception:
                pass
            self._local.conn = None
