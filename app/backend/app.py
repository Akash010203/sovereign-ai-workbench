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
from werkzeug.utils import secure_filename

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
from rag.citations import build_rag_prompt, format_citations
from rag.hybrid import HybridRetriever
from rag.fineweb_index import FineWebDiskIndex, FineWebRetriever
from rag.fineweb_lexical import FineWebLexicalRetriever
from security.offline_mode import verify_offline
from security.audit import log_event

setup_logging("backend")
log = logging.getLogger(__name__)

_GREETING_ALIASES = {"hi", "ihi", "hello", "hey", "good morning", "good afternoon", "good evening"}
_QUERY_STOP_WORDS = {
    "about", "after", "again", "also", "answer", "before", "could", "does", "from",
    "have", "help", "into", "know", "local", "more", "need", "please", "should",
    "tell", "that", "the", "their", "there", "these", "this", "what", "when", "where",
    "which", "with", "would", "your",
}


def _chat_response_kind(user_input: str) -> str:
    """Classify greetings without suppressing ordinary user questions."""
    normalized = re.sub(r"[^a-z0-9 ]", "", user_input.lower()).strip()
    if normalized in _GREETING_ALIASES:
        return "greeting"
    return "question"


def _has_local_evidence(query: str, result) -> bool:
    """Reject fuzzy vector matches that contain none of the question's subject words."""
    def normalize(word: str) -> str:
        word = word.lower().replace("calliper", "caliper")
        return word[:-1] if word.endswith("s") and len(word) > 4 else word

    terms = {
        normalize(word)
        for word in re.findall(r"[a-z0-9]+", query.lower())
        if len(word) >= 4 and word not in _QUERY_STOP_WORDS
    }
    if not terms:
        return True
    passage_terms = {
        normalize(word) for word in re.findall(r"[a-z0-9]+", result.text.lower())
    }
    return bool(terms & passage_terms)


def _extractive_local_reply(query: str, results: list) -> str:
    """Present retrieved local material without inventing an answer."""
    query_terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if len(term) >= 4]
    excerpts = []
    for result in results[:2]:
        text = re.sub(r"\s+", " ", result.text).strip()
        if text:
            lowered = text.lower().replace("calliper", "caliper")
            positions = [lowered.find(term.replace("calliper", "caliper")) for term in query_terms]
            positions = [position for position in positions if position >= 0]
            if positions:
                start = max(0, min(positions) - 150)
                end = min(len(text), start + 560)
                excerpt = text[start:end]
                if start:
                    excerpt = "…" + excerpt
                if end < len(text):
                    excerpt = excerpt.rstrip() + "…"
            else:
                excerpt = text[:420].rstrip()
            excerpts.append(f"- {excerpt}")
    if not excerpts:
        return "No readable local passage was found."
    return (
        "I found these relevant passages in your local knowledge base:\n\n"
        + "\n\n".join(excerpts)
        + "\n\nThese are retrieved local excerpts, not an answer invented by the model."
    )


def _is_low_quality_generation(reply: str) -> bool:
    """Reject obvious repetition or uncertainty from a small under-trained model."""
    lowered = reply.lower().strip()
    if not lowered or "not sure how to proceed" in lowered:
        return True
    words = re.findall(r"\w+", lowered)
    return len(words) >= 12 and len(set(words)) / len(words) < 0.38


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

    # Model registry: the project's direct, in-process MiniLLM only.
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
    # Use the tokenizer that matches the trained checkpoint.
    # Industrial 80M was trained with the 16K blended vocabulary;
    # earlier small/medium experiments used the 8K or demo vocab.
    _tok_16k  = ROOT / "tokenizer" / "vocab" / "blended_16k.json"
    _tok_8k   = ROOT / "tokenizer" / "vocab" / "bpe_8k.json"
    _tok_demo = ROOT / "tokenizer" / "vocab" / "demo_bpe_vocab.json"
    if _industrial.exists() and _tok_16k.exists():
        tok_path = str(_tok_16k)
    elif _tok_8k.exists():
        tok_path = str(_tok_8k)
    else:
        tok_path = str(_tok_demo)
    log.info("Tokenizer: %s", tok_path)
    model_registry = build_default_registry(
        minilm_checkpoint=ckpt_path,
        tokenizer_path=tok_path,
    )

    # ── Local user knowledge index ──────────────────────────────────────
    rag_index_path = Path(config.get(
        "knowledge_index", ROOT / "data" / "user_knowledge_index.json"
    ))
    knowledge_upload_dir = ROOT / "data" / "user_knowledge"
    knowledge_upload_dir.mkdir(parents=True, exist_ok=True)
    rag_index      = LocalVectorIndex()
    if rag_index_path.exists():
        rag_index.load(rag_index_path)
    rag_ingester  = DocumentIngester(rag_index)
    rag_retriever = Retriever(rag_index)
    log.info("User knowledge base loaded: %d passages", len(rag_index))

    # ── FineWeb 1.6B knowledge store (disk-backed, 5.2M chunks) ────────
    fineweb_dir = ROOT / "data" / "knowledge" / "fineweb_edu_1p6b_embed"
    fineweb_retriever = None
    lexical_retriever = None
    if fineweb_dir.exists():
        try:
            fw_index = FineWebDiskIndex(fineweb_dir)
            if fw_index.status.available:
                fineweb_retriever = FineWebRetriever(fw_index)
                log.info("FineWeb knowledge connected: %d chunks, %d-dim",
                         fw_index.status.chunks, fw_index.status.dimension)
            try:
                lexical_retriever = FineWebLexicalRetriever(fineweb_dir)
                log.info("FineWeb lexical search available.")
            except Exception:
                pass
        except Exception as exc:
            log.warning("FineWeb index load failed: %s", exc)

    # ── Hybrid retriever: local knowledge + FineWeb semantic + lexical ─
    knowledge_retriever = HybridRetriever(
        local=rag_retriever,
        fineweb=fineweb_retriever,
        lexical=lexical_retriever,
    )
    log.info("Hybrid retriever active: local=%d, fineweb=%s, lexical=%s",
             len(rag_index),
             "connected" if fineweb_retriever else "offline",
             "connected" if lexical_retriever else "offline")

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
            "knowledge_sources": len({record["source"] for record in rag_index._store.values()}),
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

        # A short greeting does not need the language model. Every substantive
        # message, including general questions, continues to generation.
        response_kind = _chat_response_kind(user_input)

        # Search local knowledge by default. The client can explicitly opt out
        # for a faster direct model response.
        rag_results = []
        model_input = user_input
        should_use_rag = use_rag is not False and len(rag_index) > 0
        if should_use_rag:
            rag_results = knowledge_retriever.retrieve(user_input, top_k=2, min_score=0.20)
            rag_results = [
                result for result in rag_results
                if _has_local_evidence(user_input, result)
            ]
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
                "Hello — I answer from the local knowledge you add. Open Knowledge base "
                "to paste text or upload documents, then ask a question about them."
            )
        elif rag_results:
            reply = _extractive_local_reply(user_input, rag_results)
        elif decision.model:
            try:
                reply = decision.model.generate(
                    model_input,
                    # Conservative decoding is more reliable for the 80M
                    # domain model than the creative defaults used by legacy
                    # demo checkpoints.
                    GenerationConfig(max_new_tokens=128, temperature=0.25, top_k=20, top_p=0.85)
                )
                if _is_low_quality_generation(reply):
                    reply = (
                        "I do not have reliable local knowledge for that question yet. Add a relevant "
                        "document in Knowledge base, then ask again."
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

    @app.route("/api/rag/upload", methods=["POST"])
    def rag_upload():
        """Store and index user-selected local files without any network upload."""
        allowed_extensions = {
            ".txt", ".md", ".csv", ".json", ".pdf", ".docx", ".xlsx", ".pptx",
            ".png", ".jpg", ".jpeg", ".webp", ".gif", ".bmp", ".tif", ".tiff",
        }
        uploaded_files = request.files.getlist("files")
        if not uploaded_files:
            return jsonify({"error": "Choose at least one document."}), 400

        added_chunks = 0
        imported_sources = []
        rejected = []
        for uploaded in uploaded_files:
            filename = secure_filename(uploaded.filename or "")
            suffix = Path(filename).suffix.lower()
            if not filename or suffix not in allowed_extensions:
                rejected.append(uploaded.filename or "unnamed file")
                continue
            destination = knowledge_upload_dir / filename
            uploaded.save(destination)
            try:
                count = rag_ingester.ingest_file(destination)
            except Exception as exc:
                rejected.append(f"{filename}: {exc}")
                continue
            added_chunks += count
            imported_sources.append({"source": filename, "chunks_added": count})

        if added_chunks:
            rag_index.save(rag_index_path)
        return jsonify({
            "chunks_added": added_chunks,
            "total_chunks": len(rag_index),
            "imported": imported_sources,
            "rejected": rejected,
        })

    @app.route("/api/rag/sources")
    def rag_sources():
        counts: dict[str, int] = {}
        for record in rag_index._store.values():
            source = record["source"]
            counts[source] = counts.get(source, 0) + 1
        sources = [
            {"source": source, "chunks": count}
            for source, count in sorted(counts.items(), key=lambda item: item[0].lower())
        ]
        return jsonify({"total_sources": len(sources), "sources": sources[:100]})

    @app.route("/api/rag/import-project-knowledge", methods=["POST"])
    def import_project_knowledge():
        """Explicitly import the earlier local index into the user knowledge base."""
        project_index_path = ROOT / "data" / "rag_index_tfidf.json"
        if not project_index_path.is_file():
            return jsonify({"error": "No compatible project knowledge index was found."}), 404

        project_index = LocalVectorIndex()
        project_index.load(project_index_path)
        expected_dimension = len(rag_retriever.embedder.embed("dimension check"))
        compatible = {
            chunk_id: record for chunk_id, record in project_index._store.items()
            if len(record.get("embedding", [])) == expected_dimension
        }
        added = sum(chunk_id not in rag_index._store for chunk_id in compatible)
        rag_index._store.update(compatible)
        rag_index.save(rag_index_path)
        return jsonify({
            "chunks_added": added,
            "total_chunks": len(rag_index),
            "source_index": project_index_path.name,
        })

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
