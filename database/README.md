# database/ (reserved)

Empty until **Phase 17** (see `docs/ARCHITECTURE.md` §11).

Planned contents: `schema.sql`, `db.py`, `repositories/` — SQLite for
application state (conversations, tasks, tool calls, agent state,
documents, audit records, model registry, training experiments). Kept
strictly separate from ML logic; SQLite is **not** the neural network.
