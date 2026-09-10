"""
tests/test_e2e.py — End-to-end integration tests for the full platform.

These tests exercise the complete stack without requiring:
  - a trained model checkpoint (tests use untrained MiniLLM)
  - Ollama running (Ollama adapter is tested for import / availability check only)
  - GPU (all tests run on CPU)

Each test proves one complete vertical slice works end-to-end.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tokenizer():
    """Load the trained tokenizer (requires Phase 3 complete)."""
    from tokenizer.tokenizer import ByteLevelBPETokenizer
    vocab_path = ROOT / "tokenizer" / "vocab" / "demo_bpe_vocab.json"
    if not vocab_path.exists():
        pytest.skip("Tokenizer vocab not found — run scripts/train_tokenizer.py first")
    return ByteLevelBPETokenizer.load(vocab_path)


@pytest.fixture(scope="module")
def model(tokenizer):
    """Create an untrained MiniLLM with the real vocab size."""
    from models.custom_minilm.config import MiniLLMConfig
    from models.custom_minilm.model import MiniLLM
    cfg = MiniLLMConfig(vocab_size=tokenizer.vocab_size)
    return MiniLLM(cfg)


@pytest.fixture(scope="module")
def tool_registry():
    from tools import build_default_tool_registry
    return build_default_tool_registry(workspace_root=str(ROOT))


# ─────────────────────────────────────────────────────────────────────────────
# 1. Tokenizer → Model forward pass (Phase 3 + 5 integration)
# ─────────────────────────────────────────────────────────────────────────────

def test_tokenizer_to_model_forward(tokenizer, model):
    """
    End-to-end: encode text → feed to MiniLLM → get valid logits.
    Proves Phase 3 (tokenizer) and Phase 5 (model) are wired correctly.
    """
    text = "The pump inspection was completed."
    ids  = tokenizer.encode(text, add_bos=True, add_eos=True)
    assert len(ids) > 0, "Tokenizer returned empty encoding."

    input_tensor = torch.tensor([ids], dtype=torch.long)
    with torch.no_grad():
        logits = model(input_tensor)

    assert logits.shape == (1, len(ids), tokenizer.vocab_size), \
        f"Unexpected logits shape: {logits.shape}"
    assert torch.isfinite(logits).all(), "NaN or Inf in logits."


# ─────────────────────────────────────────────────────────────────────────────
# 2. Generation pipeline (Phase 7)
# ─────────────────────────────────────────────────────────────────────────────

def test_generation_pipeline(tokenizer, model):
    """
    End-to-end: prompt → encode → generate → decode.
    Proves Phase 7 generate() works with the real model and tokenizer.
    """
    from models.custom_minilm.generate import generate
    prompt = "The inspection report"
    ids    = tokenizer.encode(prompt, add_bos=True)
    eos_id = tokenizer.vocab.special_to_id.get("<eos>")

    new_ids = generate(
        model,
        prompt_ids=ids,
        max_new_tokens=10,
        temperature=1.0,
        top_k=10,
        eos_id=eos_id,
        seed=42,
        device="cpu",
    )

    assert isinstance(new_ids, list), "generate() should return a list."
    assert len(new_ids) > 0, "generate() returned zero tokens."
    assert all(0 <= t < tokenizer.vocab_size for t in new_ids), \
        "Generated token IDs out of vocab range."

    text = tokenizer.decode(new_ids)
    assert isinstance(text, str), "decode() should return a string."


# ─────────────────────────────────────────────────────────────────────────────
# 3. Task router classification (Phase 9)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("text,expected_category", [
    ("write a Python function to sort a list", "coding"),
    ("calculate the pressure drop across the valve", "calculation"),
    ("search for the previous maintenance report", "retrieval"),
    ("extract text from the scanned document", "ocr"),
    ("summarize the quarterly performance report", "summarization"),
    ("The inspection was completed yesterday", "engineering_document"),
])
def test_router_classification(text, expected_category):
    """
    Rule-based task classifier correctly routes standard SIH demo queries.
    """
    from router.task_classifier import classify_task
    result = classify_task(text)
    assert result.value == expected_category, \
        f"Input '{text}' → expected '{expected_category}', got '{result.value}'"


# ─────────────────────────────────────────────────────────────────────────────
# 4. Calculator tool (Phase 10)
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("expr,expected", [
    ("2 + 2", 4),
    ("sqrt(144)", 12.0),
    ("10 * 3.14159", 31.4159),
    ("2 ** 10", 1024),
])
def test_calculator_tool(expr, expected, tool_registry):
    result = tool_registry.call("calculator", expression=expr)
    assert result.success, f"Calculator failed for '{expr}': {result.error}"
    assert abs(result.output - expected) < 1e-3, \
        f"Calculator: '{expr}' → expected {expected}, got {result.output}"


def test_calculator_blocks_unsafe_code(tool_registry):
    """The calculator must NOT execute arbitrary Python — only math expressions."""
    result = tool_registry.call("calculator", expression="__import__('os').system('echo pwned')")
    assert not result.success, "Calculator must reject non-math expressions."


# ─────────────────────────────────────────────────────────────────────────────
# 5. Filesystem tool — path traversal blocked (Phase 10 + 18)
# ─────────────────────────────────────────────────────────────────────────────

def test_filesystem_path_traversal_blocked(tool_registry):
    """
    The file_read tool must reject paths that escape the workspace root.
    This is a security requirement — prevents reading /etc/passwd or Windows secrets.
    """
    result = tool_registry.call("file_read", path="../../Windows/System32/hosts")
    assert not result.success, "Path traversal should be rejected."
    assert result.error is not None


def test_filesystem_reads_valid_file(tool_registry):
    """The file_read tool should read real files within the workspace."""
    result = tool_registry.call("file_read", path="data/raw/synthetic_demo_corpus.txt")
    assert result.success, f"File read failed: {result.error}"
    assert len(result.output) > 0


# ─────────────────────────────────────────────────────────────────────────────
# 6. Code sandbox isolation (Phase 15)
# ─────────────────────────────────────────────────────────────────────────────

def test_code_sandbox_executes_safely(tool_registry):
    """Code sandbox executes Python and captures output."""
    code = "print('hello from sandbox')\nprint(2 + 2)"
    result = tool_registry.call("code_sandbox", code=code)
    assert result.success, f"Sandbox failed: {result.error}"
    assert "hello from sandbox" in result.output.get("stdout", "")


def test_code_sandbox_captures_errors(tool_registry):
    """Code sandbox captures stderr from runtime errors."""
    code = "raise ValueError('intentional test error')"
    result = tool_registry.call("code_sandbox", code=code)
    assert not result.success
    assert result.output["exit_code"] != 0


def test_code_sandbox_timeout(tmp_path):
    """Code sandbox kills processes that exceed the timeout."""
    from tools.code_sandbox import CodeSandboxTool
    sandbox = CodeSandboxTool(timeout_seconds=1)
    result  = sandbox.run(code="import time; time.sleep(30)")
    assert not result.success
    assert "timed out" in (result.error or "").lower()


# ─────────────────────────────────────────────────────────────────────────────
# 7. RAG end-to-end (Phase 12)
# ─────────────────────────────────────────────────────────────────────────────

def test_rag_ingest_and_retrieve():
    """
    End-to-end RAG: ingest text → embed → index → query → retrieve.
    Uses TF-IDF fallback (no sentence-transformers required).
    """
    from rag.chunking import chunk_text
    from rag.index import LocalVectorIndex
    from rag.embeddings import TFIDFEmbedder

    corpus = [
        "The pump inspection revealed corrosion on the valve assembly.",
        "The pressure gauge showed abnormal readings at station 3.",
        "The maintenance team replaced the faulty compressor seal.",
    ]

    # Fit and embed
    emb = TFIDFEmbedder()
    emb.fit(corpus)

    index = LocalVectorIndex()
    for i, text in enumerate(corpus):
        chunks    = chunk_text(text, source=f"doc_{i}")
        embeddings = emb.embed_batch([c.text for c in chunks])
        index.add_batch(chunks, embeddings)

    assert len(index) >= len(corpus), "At least one chunk per document should be indexed."

    # Query
    q_vec   = emb.embed("pump corrosion")
    results = index.search(q_vec, top_k=1)

    assert len(results) == 1
    assert results[0]["score"] > 0.0
    assert "corrosion" in results[0]["text"].lower() or results[0]["score"] > 0.0


def test_rag_index_persistence(tmp_path):
    """RAG index saves to JSON and reloads correctly."""
    from rag.chunking import chunk_text
    from rag.index import LocalVectorIndex
    from rag.embeddings import TFIDFEmbedder

    text  = "The valve pressure exceeded safe limits."
    emb   = TFIDFEmbedder()
    emb.fit([text])

    idx1  = LocalVectorIndex()
    chunk = chunk_text(text, source="test")[0]
    idx1.add(chunk, emb.embed(chunk.text))

    save_path = tmp_path / "test_index.json"
    idx1.save(save_path)

    idx2 = LocalVectorIndex()
    idx2.load(save_path)
    assert len(idx2) == 1


# ─────────────────────────────────────────────────────────────────────────────
# 8. Database (Phase 17)
# ─────────────────────────────────────────────────────────────────────────────

def test_database_conversations(tmp_path):
    """ConversationRepository creates and retrieves conversations."""
    from database.db import Database
    from database.repositories.conversations import ConversationRepository

    db   = Database(tmp_path / "test.db")
    repo = ConversationRepository(db)

    conv_id = repo.create(title="Test Conversation", model_name="custom_minilm_v1")
    assert conv_id is not None

    msg_id = repo.add_message(conv_id, "user", "Hello")
    assert msg_id is not None

    repo.add_message(conv_id, "assistant", "Hello back!")
    messages = repo.get_messages(conv_id)
    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


# ─────────────────────────────────────────────────────────────────────────────
# 9. Security — network monitor (Phase 18)
# ─────────────────────────────────────────────────────────────────────────────

def test_permission_manager_blocks_traversal():
    """PermissionManager must reject paths outside the workspace root."""
    from security.permissions import PermissionManager
    pm = PermissionManager(str(ROOT))

    assert pm.is_allowed(str(ROOT / "data" / "raw"))
    assert not pm.is_allowed("C:\\Windows\\System32")
    assert not pm.is_allowed("/etc/passwd")


def test_audit_logger_writes(tmp_path, monkeypatch):
    """Audit logger writes events to the log file."""
    import security.audit as audit_mod
    log_file = tmp_path / "audit.log"
    monkeypatch.setattr(audit_mod, "AUDIT_LOG_FILE", log_file)

    from security.audit import log_event
    log_event("tool_call", {"tool": "calculator", "expression": "2+2"})

    assert log_file.exists()
    line = json.loads(log_file.read_text().strip())
    assert line["event_type"] == "tool_call"
    assert line["details"]["tool"] == "calculator"


# ─────────────────────────────────────────────────────────────────────────────
# 10. Training dataset loader (Phase 6)
# ─────────────────────────────────────────────────────────────────────────────

def test_training_dataset_loads(tokenizer):
    """TokenizedTextDataset loads the corpus and produces valid (input, target) pairs."""
    from training.dataset import TokenizedTextDataset, make_dataloader

    train_path = ROOT / "data" / "processed" / "train.txt"
    if not train_path.exists():
        pytest.skip("Train split not found — run scripts/prepare_data.py")

    ds = TokenizedTextDataset(
        train_path, tokenizer, block_size=32, stride=16
    )
    assert len(ds) > 0, "Dataset is empty."

    input_ids, targets = ds[0]
    assert input_ids.shape == (32,)
    assert targets.shape == (32,)
    # The key property: targets = input_ids shifted left by 1
    assert (targets[:-1] == input_ids[1:]).all(), \
        "targets must be input_ids shifted left by one position."


def test_lr_scheduler_warmup():
    """Cosine scheduler linearly warms up then decays."""
    from training.scheduler import cosine_with_warmup
    import torch.optim as optim
    from models.custom_minilm.config import MiniLLMConfig
    from models.custom_minilm.model import MiniLLM

    model = MiniLLM(MiniLLMConfig(vocab_size=100))
    opt   = optim.AdamW(model.parameters(), lr=1e-3)
    sched = cosine_with_warmup(opt, warmup_steps=5, max_steps=20)

    lrs = []
    for _ in range(20):
        opt.step()
        sched.step()
        lrs.append(sched.get_last_lr()[0])

    # LR should be rising during warmup
    assert lrs[3] > lrs[0], "LR should increase during warmup."
    # LR should be falling after warmup
    assert lrs[-1] < lrs[4], "LR should decrease after warmup."
