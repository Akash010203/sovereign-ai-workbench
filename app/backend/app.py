"""
app/backend/app.py — SovereignAI Flask backend.

LOCAL BACKEND — NOT FastAPI (as required by the build contract).
Flask is lightweight, no external runtime dependencies.
Runs on http://localhost:5000 — NOT exposed to the internet.

API ROUTES
----------
GET  /api/status              → system status (models, offline check)
GET  /api/models              → list all registered models
POST /api/chat                → single-turn chat completion
POST /api/agent               → full agentic task execution
GET  /api/tasks               → list recent tasks
GET  /api/tasks/<id>          → get one task
POST /api/rag/ingest          → ingest a document into the knowledge base
POST /api/rag/search          → search the knowledge base
GET  /api/security/offline    → run offline verification
GET  /api/audit               → recent audit log entries
"""
from __future__ import annotations

import json
import logging
import os
import re
import sys
from pathlib import Path

# ── Allow imports from project root ───────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

from core.config import get_settings, ensure_directories
from core.logging_setup import setup_logging
from database.db import Database
from database.repositories.conversations import ConversationRepository, TaskRepository
from models.registry import build_default_registry
from tools import build_default_tool_registry
from agents.agent import Agent
from rag.index import LocalVectorIndex
from rag.ingest import DocumentIngester
from rag.retriever import Retriever
from rag.fineweb_index import FineWebDiskIndex, FineWebRetriever
from rag.hybrid import HybridRetriever
from rag.citations import build_rag_prompt, format_citations
from security.offline_mode import verify_offline
from security.audit import log_event

setup_logging("backend")
log = logging.getLogger(__name__)

_DOMAIN_TERMS = {
    "alarm", "bearing", "compressor", "equipment", "fault", "inspection",
    "maintenance", "manual", "motor", "pipeline", "pressure", "procedure",
    "pump", "report", "sensor", "sop", "temperature", "valve", "vibration",
    "work order", "work instruction",
}
_GREETING_ALIASES = {"hi", "ihi", "hello", "hey", "good morning", "good afternoon", "good evening"}
_VAGUE_MESSAGES = {"what", "what is", "what is it", "help", "help me", "why", "how"}


def _chat_response_kind(user_input: str) -> str:
    """Keep a specialised local model within its demonstrated scope."""
    normalized = re.sub(r"[^a-z0-9 ]", "", user_input.lower()).strip()
    if normalized in _GREETING_ALIASES:
        return "greeting"
    if normalized in _VAGUE_MESSAGES or len(normalized) <= 3:
        return "needs_detail"
    return "domain" if any(term in normalized for term in _DOMAIN_TERMS) else "out_of_scope"


def create_app(config: dict = None) -> Flask:
    """Application factory — create and configure the Flask app."""

    app = Flask(
        __name__,
        static_folder=str(ROOT / "app" / "frontend"),
        static_url_path="",
    )
    CORS(app)

    config = config or {}
    ensure_directories()

    # ── Initialize subsystems ──────────────────────────────────────────
    db        = Database()
    conv_repo = ConversationRepository(db)
    task_repo = TaskRepository(db)

    # Model registry (MiniLLM + Ollama)
    # Priority: current industrial 80M model → legacy checkpoints.
    _industrial = ROOT / "checkpoints" / "industrial_80m_best.pt"
    _8k_best    = ROOT / "checkpoints" / "best_8k.pt"
    _finetuned  = ROOT / "checkpoints" / "finetuned" / "finetune_best.pt"
    _pretrained = ROOT / "checkpoints" / "best.pt"
    if _industrial.exists():
        _default_ckpt = str(_industrial)
        log.info("Using industrial 80M checkpoint: %s", _default_ckpt)
    elif _8k_best.exists():
        _default_ckpt = str(_8k_best)
        log.info("Using 8k-vocab checkpoint: %s", _default_ckpt)
    elif _finetuned.exists():
        _default_ckpt = str(_finetuned)
        log.info("Using fine-tuned checkpoint: %s", _default_ckpt)
    else:
        _default_ckpt = str(_pretrained)
        log.info("Using pretrained checkpoint: %s", _default_ckpt)
    ckpt_path = config.get("minilm_checkpoint", _default_ckpt)
    # tokenizer_path is now read from inside the checkpoint automatically
    tok_path  = str(ROOT / "tokenizer" / "vocab" / "demo_bpe_vocab.json")
    model_registry = build_default_registry(
        minilm_checkpoint=ckpt_path,
        tokenizer_path=tok_path,
        ollama_models=config.get("ollama_models", []),  # empty = no Ollama
    )

    # RAG (must be initialized before tools — RagSearchTool wraps the retriever)
    rag_index_path = ROOT / "data" / "rag_index.json"
    rag_index      = LocalVectorIndex()
    if rag_index_path.exists():
        rag_index.load(rag_index_path)
    rag_ingester  = DocumentIngester(rag_index)
    rag_retriever = Retriever(rag_index)
    fineweb_index = FineWebDiskIndex(ROOT / "data" / "knowledge" / "fineweb_edu_1p6b_embed")
    fineweb_retriever = (
        FineWebRetriever(fineweb_index, embedder=rag_retriever.embedder)
        if fineweb_index.status.available else None
    )
    knowledge_retriever = HybridRetriever(rag_retriever, fineweb_retriever)
    if fineweb_retriever:
        log.info("FineWeb knowledge enabled: %d chunks via %s search", len(fineweb_index), fineweb_index.status.mode)
    else:
        log.info("FineWeb knowledge unavailable: %s", fineweb_index.status.detail)

    # Tool registry (pass retriever so agent can call rag_search via tools)
    tool_registry = build_default_tool_registry(
        workspace_root=str(ROOT),
        retriever=knowledge_retriever,
    )

    # Agent
    agent = Agent(model_registry, tool_registry)

    # ── Global error handler — always return JSON, never HTML ────────
    from werkzeug.exceptions import HTTPException

    @app.errorhandler(Exception)
    def handle_any_exception(e):
        if isinstance(e, HTTPException):
            return jsonify({"error": e.description}), e.code
        log.error("Unhandled exception in route: %s", e, exc_info=True)
        return jsonify({"error": str(e)}), 500

    # ── Routes ────────────────────────────────────────────────────────

    # Serve frontend
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder, "index.html")

    # ── Status ────────────────────────────────────────────────────────
    @app.route("/api/status")
    def status():
        offline = verify_offline()
        models  = model_registry.all_info()
        return jsonify({
            "status":   "running",
            "offline":  offline,
            "models":   models,
            "rag_docs": len(rag_index),
            "fineweb": {
                "available": fineweb_index.status.available,
                "chunks": len(fineweb_index),
                "dimension": fineweb_index.status.dimension,
                "mode": fineweb_index.status.mode,
                "detail": fineweb_index.status.detail,
            },
        })

    # ── Models ────────────────────────────────────────────────────────
    @app.route("/api/models")
    def list_models():
        return jsonify(model_registry.all_info())

    # ── Chat ──────────────────────────────────────────────────────────
    @app.route("/api/chat", methods=["POST"])
    def chat():
        data        = request.json or {}
        user_input  = data.get("message", "").strip()
        model_name  = data.get("model", "")
        conv_id     = data.get("conversation_id", "")
        # ``None`` means the client did not choose; false explicitly opts out.
        use_rag     = data.get("use_rag")

        if not user_input:
            return jsonify({"error": "Empty message."}), 400

        # Get or create conversation
        if not conv_id:
            conv_id = conv_repo.create(title=user_input[:60], model_name=model_name)
        conv_repo.add_message(conv_id, "user", user_input)

        # A short greeting does not need the language model.  This prevents a
        # small domain model from turning "hi" into an unrelated technical
        # explanation, while keeping all substantive requests model-generated.
        response_kind = _chat_response_kind(user_input)

        # Retrieve only when explicitly requested. This keeps normal chat
        # lightweight and lets the client show source provenance separately
        # instead of trusting a small generative model to format citations.
        rag_results = []
        model_input = user_input
        # Use RAG only when the user explicitly enables it in the UI. This
        # prevents irrelevant chunks from overwhelming the 256-token context.
        should_use_rag = use_rag is True and (len(rag_index) > 0 or len(fineweb_index) > 0)
        if should_use_rag:
            rag_results = knowledge_retriever.retrieve(user_input, top_k=2, min_score=0.20)
            if rag_results:
                # Leave room for the question and answer in the 256-token
                # context; oversized RAG text was truncated before generation.
                model_input = build_rag_prompt(
                    user_input, rag_results, max_context_chars=350
                )

        # Route and generate
        from router.router import TaskRouter
        from models.adapters.base import GenerationConfig
        decision = TaskRouter(model_registry).route(user_input)

        reply = "[No model available]"
        if response_kind == "greeting":
            reply = (
                "Hello — I’m SovereignAI, your local maintenance and engineering assistant. "
                "Ask me about an equipment issue, a work instruction, or your local knowledge base."
            )
        elif response_kind == "needs_detail":
            reply = (
                "Please give me a specific maintenance or engineering question. For example: "
                "‘Pump P-101 has high drive-end vibration—what should I inspect first?’"
            )
        elif response_kind == "out_of_scope" and not rag_results:
            reply = (
                "This local model is specialised for maintenance and engineering, not general chat. "
                "Ask about equipment, a fault, an inspection, a work order, or enable ‘Use knowledge’ "
                "to search your local documents."
            )
        elif decision.model:
            try:
                reply = decision.model.generate(
                    model_input,
                    # Conservative decoding is more reliable for the 80M
                    # domain model than the creative defaults used by legacy
                    # demo checkpoints.
                    GenerationConfig(max_new_tokens=128, temperature=0.25, top_k=20, top_p=0.85)
                )
            except Exception as exc:
                reply = f"[Model error: {exc}]"

        conv_repo.add_message(conv_id, "assistant", reply)
        log_event("model_call", {
            "model": decision.model_name, "category": decision.category.value,
            "input_len": len(user_input), "reply_len": len(reply),
        })

        return jsonify({
            "reply":           reply,
            "conversation_id": conv_id,
            "model_used":      decision.model_name,
            "category":        decision.category.value,
            "routing_reason":  decision.reason,
            "rag_used":        bool(rag_results),
            "sources": [
                {"source": r.source, "chunk_id": r.chunk_id, "score": r.score}
                for r in rag_results
            ],
        })

    # ── Conversations ──────────────────────────────────────────────────
    @app.route("/api/conversations", methods=["GET"])
    def list_conversations():
        return jsonify(conv_repo.list_conversations())

    @app.route("/api/conversations/<conv_id>", methods=["GET"])
    def get_conversation(conv_id):
        messages = conv_repo.get_messages(conv_id)
        return jsonify({"conversation_id": conv_id, "messages": messages})

    @app.route("/api/conversations/<conv_id>", methods=["DELETE"])
    def delete_conversation(conv_id):
        conv_repo.delete_conversation(conv_id)
        return jsonify({"deleted": True, "conversation_id": conv_id})

    # ── Agent ─────────────────────────────────────────────────────────
    @app.route("/api/agent", methods=["POST"])
    def run_agent():
        data       = request.json or {}
        user_input = data.get("task", "").strip()
        if not user_input:
            return jsonify({"error": "Empty task."}), 400

        task_state = agent.run(user_input)
        task_repo.save_task(task_state)
        log_event("agent_task", task_state.to_dict())

        return jsonify(task_state.to_dict())

    # ── Tasks ─────────────────────────────────────────────────────────
    @app.route("/api/tasks")
    def list_tasks():
        return jsonify(task_repo.list_tasks(limit=30))

    @app.route("/api/tasks/<task_id>")
    def get_task(task_id):
        task = task_repo.get_task(task_id)
        if not task:
            return jsonify({"error": "Not found"}), 404
        return jsonify(task)

    # ── RAG ───────────────────────────────────────────────────────────
    @app.route("/api/rag/ingest", methods=["POST"])
    def rag_ingest():
        """Ingest an uploaded file or a text snippet into the knowledge base."""
        data   = request.json or {}
        text   = data.get("text", "")
        source = data.get("source", "manual_input")
        if not text:
            return jsonify({"error": "No text provided."}), 400

        n_chunks = rag_ingester.ingest_text(text, source=source)
        rag_index.save(rag_index_path)
        return jsonify({"chunks_added": n_chunks, "total_chunks": len(rag_index)})

    @app.route("/api/rag/search", methods=["POST"])
    def rag_search():
        data  = request.json or {}
        query = data.get("query", "").strip()
        top_k = int(data.get("top_k", 5))
        if not query:
            return jsonify({"error": "Empty query."}), 400

        results = knowledge_retriever.retrieve(query, top_k=top_k)
        return jsonify({
            "results":   [{"source": r.source, "text": r.text, "score": r.score,
                           "citation": r.citation()} for r in results],
            "citations": format_citations(results),
        })

    # ── Security ──────────────────────────────────────────────────────
    @app.route("/api/security/offline")
    def offline_check():
        return jsonify(verify_offline())

    # ── Audit log ─────────────────────────────────────────────────────
    @app.route("/api/audit")
    def audit_log():
        audit_path = ROOT / "logs" / "audit.log"
        if not audit_path.exists():
            return jsonify([])
        lines = audit_path.read_text(encoding="utf-8").strip().splitlines()
        events = []
        for line in lines[-50:]:  # last 50 events
            try:
                events.append(json.loads(line))
            except Exception:
                pass
        return jsonify(events)

    return app


if __name__ == "__main__":
    port = int(os.environ.get("SOVEREIGN_PORT", 5000))
    log.info("Starting SovereignAI backend on http://localhost:%d", port)
    app = create_app()
    app.run(host="127.0.0.1", port=port, debug=False)
