# app/ (reserved)

Empty until **Phases 19-20** (see `docs/ARCHITECTURE.md` §11).

- `backend/` — the local Flask backend (project rules explicitly
  forbid FastAPI here) exposing chat, task execution, file upload,
  agent execution, status, logs, artifacts, model selection, and
  knowledge search to the UI.
- `frontend/` — the local AI-workbench web UI: Chat, Task/Agent
  execution, Uploaded documents, Knowledge base, Generated artifacts,
  Model selection/status, Security/offline status, Execution logs.
