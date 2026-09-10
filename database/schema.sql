-- database/schema.sql
-- SovereignAI application data schema (SQLite)
-- This database stores ONLY application state — conversations, tasks,
-- audit logs, model registry.  It is NOT the neural network.

PRAGMA journal_mode=WAL;
PRAGMA foreign_keys=ON;

-- ── Conversations ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
    title       TEXT,
    model_name  TEXT
);

-- ── Messages ──────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    role            TEXT NOT NULL CHECK (role IN ('user','assistant','tool','system')),
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    metadata        TEXT DEFAULT '{}'   -- JSON blob
);

-- ── Agent Tasks ───────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS tasks (
    id              TEXT PRIMARY KEY,
    conversation_id TEXT REFERENCES conversations(id),
    user_input      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending',
    model_name      TEXT,
    category        TEXT,
    final_answer    TEXT,
    error           TEXT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at      TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Task Steps ────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS task_steps (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT NOT NULL REFERENCES tasks(id) ON DELETE CASCADE,
    step_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    tool_name   TEXT,
    tool_args   TEXT DEFAULT '{}',   -- JSON
    result      TEXT,
    success     INTEGER,             -- 0 or 1
    error       TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Artifacts (generated files) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS artifacts (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    task_id     TEXT REFERENCES tasks(id),
    name        TEXT NOT NULL,
    path        TEXT NOT NULL,
    type        TEXT,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Documents (knowledge base) ────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    filename    TEXT NOT NULL,
    source_path TEXT,
    status      TEXT DEFAULT 'pending',   -- pending | indexed | error
    chunks      INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL DEFAULT (datetime('now'))
);

-- ── Audit log (tool calls + security events) ─────────────────────────────
CREATE TABLE IF NOT EXISTS audit_log (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    event_type  TEXT NOT NULL,   -- tool_call | security | model_call | system
    details     TEXT DEFAULT '{}',   -- JSON blob
    task_id     TEXT
);

-- ── Model registry ────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS model_registry (
    name        TEXT PRIMARY KEY,
    model_type  TEXT,
    is_custom   INTEGER DEFAULT 0,
    params      INTEGER DEFAULT 0,
    notes       TEXT,
    registered_at TEXT DEFAULT (datetime('now'))
);

-- ── Training experiments ──────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS training_experiments (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    max_steps       INTEGER,
    final_train_loss REAL,
    best_val_loss   REAL,
    checkpoint_path TEXT,
    device          TEXT,
    completed_at    TEXT DEFAULT (datetime('now'))
);
