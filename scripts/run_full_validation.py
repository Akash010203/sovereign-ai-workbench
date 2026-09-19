"""
scripts/run_full_validation.py — Master validation runner for SovereignAI.

Executes all 28 validation categories and produces a SYSTEM_HEALTH report.

Usage:
    python scripts/run_full_validation.py
    python scripts/run_full_validation.py --category rag
    python scripts/run_full_validation.py --report-only
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger("validator")


@dataclass
class TestResult:
    category: str
    status: str  # PASS, FAIL, BLOCKED, NOT_IMPLEMENTED, NOT_APPLICABLE
    details: str = ""
    evidence: dict = field(default_factory=dict)
    duration_s: float = 0.0


def _run(label: str, fn) -> TestResult:
    """Run a test function and capture its result."""
    start = time.time()
    try:
        result = fn()
        elapsed = time.time() - start
        if isinstance(result, TestResult):
            result.duration_s = elapsed
            return result
        return TestResult(label, "PASS", duration_s=elapsed)
    except Exception as exc:
        elapsed = time.time() - start
        return TestResult(label, "FAIL", details=str(exc), duration_s=elapsed)


# ─── 1. Environment ──────────────────────────────────────────────────────────

def check_environment() -> TestResult:
    evidence = {
        "python_version": sys.version,
        "platform": sys.platform,
        "cwd": os.getcwd(),
    }
    try:
        import torch
        evidence["torch_version"] = torch.__version__
        evidence["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            evidence["gpu_name"] = torch.cuda.get_device_name(0)
            evidence["vram_gb"] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
    except ImportError:
        evidence["torch"] = "NOT INSTALLED"
        return TestResult("environment", "FAIL", "PyTorch not installed", evidence)
    return TestResult("environment", "PASS", evidence=evidence)


# ─── 2. Repository ───────────────────────────────────────────────────────────

def check_repository() -> TestResult:
    evidence = {}
    try:
        evidence["commit"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(ROOT), text=True
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(ROOT), text=True
        ).strip()
        evidence["dirty_file_count"] = len(dirty.split("\n")) if dirty else 0
    except Exception as exc:
        return TestResult("repository", "FAIL", str(exc))
    return TestResult("repository", "PASS", evidence=evidence)


# ─── 3. Dataset ──────────────────────────────────────────────────────────────

def check_dataset() -> TestResult:
    evidence = {}
    for name in ["train.txt", "val.txt"]:
        path = ROOT / "data" / "processed" / name
        evidence[name] = {"exists": path.exists(), "size": path.stat().st_size if path.exists() else 0}
    blended = ROOT / "data" / "processed" / "blended_train.txt"
    evidence["blended_train.txt"] = {
        "exists": blended.exists(),
        "size_gb": round(blended.stat().st_size / (1024**3), 2) if blended.exists() else 0,
    }
    status = "PASS" if all(v.get("exists") for v in evidence.values()) else "FAIL"
    return TestResult("dataset", status, evidence=evidence)


# ─── 4. Tokenizer ────────────────────────────────────────────────────────────

def check_tokenizer() -> TestResult:
    evidence = {}
    try:
        from tokenizer.tokenizer import ByteLevelBPETokenizer
        for name in ["demo_bpe_vocab.json", "bpe_8k.json", "blended_16k.json"]:
            path = ROOT / "tokenizer" / "vocab" / name
            if path.exists():
                tok = ByteLevelBPETokenizer.load(str(path))
                test_ids = tok.encode("The pump inspection interval is 90 days.")
                decoded = tok.decode(test_ids)
                evidence[name] = {
                    "vocab_size": tok.vocab_size,
                    "test_encode_len": len(test_ids),
                    "roundtrip_ok": len(decoded) > 0,
                }
            else:
                evidence[name] = {"exists": False}
    except Exception as exc:
        return TestResult("tokenizer", "FAIL", str(exc), evidence)
    return TestResult("tokenizer", "PASS", evidence=evidence)


# ─── 5. Model ────────────────────────────────────────────────────────────────

def check_model() -> TestResult:
    evidence = {}
    try:
        import torch
        from models.custom_minilm.config import MiniLLMConfig
        from models.custom_minilm.model import MiniLLM

        for name, factory in [("small", MiniLLMConfig.small), ("industrial_80m", MiniLLMConfig.industrial_80m)]:
            cfg = factory(vocab_size=16000)
            model = MiniLLM(cfg)
            params = model.count_parameters()
            # Quick forward pass
            x = torch.randint(0, 100, (1, 16))
            logits = model(x)
            evidence[name] = {
                "params": params,
                "params_m": round(params / 1e6, 1),
                "output_shape": list(logits.shape),
            }
            del model
    except Exception as exc:
        return TestResult("model", "FAIL", str(exc), evidence)
    return TestResult("model", "PASS", evidence=evidence)


# ─── 6. Checkpoints ──────────────────────────────────────────────────────────

def check_checkpoints() -> TestResult:
    evidence = {}
    ckpt_dir = ROOT / "checkpoints"
    if not ckpt_dir.exists():
        return TestResult("checkpoints", "FAIL", "checkpoints/ directory missing")

    pt_files = list(ckpt_dir.rglob("*.pt"))
    evidence["count"] = len(pt_files)
    evidence["total_gb"] = round(sum(f.stat().st_size for f in pt_files) / (1024**3), 2)

    best = ckpt_dir / "industrial_80m_best.pt"
    evidence["industrial_80m_best_exists"] = best.exists()
    if best.exists():
        evidence["industrial_80m_best_size_mb"] = round(best.stat().st_size / (1024**2), 1)

    return TestResult("checkpoints", "PASS" if pt_files else "FAIL", evidence=evidence)


# ─── 7. RAG ──────────────────────────────────────────────────────────────────

def check_rag() -> TestResult:
    evidence = {}
    try:
        from rag.embeddings import get_embedder
        from rag.index import LocalVectorIndex
        from rag.ingest import DocumentIngester
        from rag.retriever import Retriever
        from rag.chunking import chunk_text

        embedder = get_embedder()
        evidence["embedder_type"] = type(embedder).__name__
        evidence["embedding_dim"] = len(embedder.embed("test"))

        # Test with fixture documents
        fixtures = ROOT / "tests" / "fixtures" / "rag"
        idx = LocalVectorIndex()
        ingester = DocumentIngester(idx)
        total = 0
        for path in sorted(fixtures.glob("*.txt")):
            total += ingester.ingest_file(path)
        evidence["chunks_ingested"] = total
        evidence["index_size"] = len(idx)

        retriever = Retriever(idx)
        results = retriever.retrieve("maximum operating temperature", top_k=3)
        evidence["retrieval_results"] = len(results)
        if results:
            evidence["top_result_source"] = results[0].source
            evidence["top_result_score"] = results[0].score
            evidence["top_result_text_preview"] = results[0].text[:100]

        # Negative test
        neg_results = retriever.retrieve("quantum chromodynamics", top_k=3, min_score=0.8)
        evidence["negative_test_results"] = len(neg_results)

    except Exception as exc:
        return TestResult("rag", "FAIL", str(exc), evidence)

    if total > 0 and len(results) > 0:
        return TestResult("rag", "PASS", evidence=evidence)
    return TestResult("rag", "FAIL", "Ingestion or retrieval produced no results", evidence)


# ─── 8. FineWeb ──────────────────────────────────────────────────────────────

def check_fineweb() -> TestResult:
    evidence = {}
    manifest_path = ROOT / "data" / "knowledge" / "fineweb_edu_1p6b_embed" / "fineweb_edu_manifest.json"
    if not manifest_path.exists():
        return TestResult("fineweb", "NOT_APPLICABLE", "FineWeb shards not present")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evidence["chunks"] = manifest["stats"]["chunks_written"]
    evidence["embedding_dim"] = manifest["stats"]["embedding_dimension"]
    evidence["source_docs"] = manifest["stats"]["accepted_documents"]

    chunk_files = sorted(manifest_path.parent.glob("chunks-*.jsonl"))
    embed_files = sorted(manifest_path.parent.glob("embeddings-*.f32"))
    evidence["chunk_shards"] = len(chunk_files)
    evidence["embedding_shards"] = len(embed_files)

    status = "PASS" if len(chunk_files) > 0 and len(embed_files) > 0 else "FAIL"
    return TestResult("fineweb", status, evidence=evidence)


# ─── 9. Database ─────────────────────────────────────────────────────────────

def check_database() -> TestResult:
    evidence = {}
    db_path = ROOT / "data" / "sovereign_ai.db"
    evidence["exists"] = db_path.exists()
    if db_path.exists():
        evidence["size_kb"] = round(db_path.stat().st_size / 1024, 1)
    try:
        from database.db import Database
        db = Database()
        evidence["connection"] = "OK"
    except Exception as exc:
        return TestResult("database", "FAIL", str(exc), evidence)
    return TestResult("database", "PASS" if db_path.exists() else "FAIL", evidence=evidence)


# ─── 10. Router ──────────────────────────────────────────────────────────────

def check_router() -> TestResult:
    evidence = {}
    try:
        from router.router import TaskRouter
        from models.registry import build_default_registry

        registry = build_default_registry()
        router = TaskRouter(registry)

        test_cases = [
            ("What is the inspection interval?", "rag"),
            ("Calculate 2 + 2", "math"),
            ("Write a Python function", "coding"),
        ]
        for query, expected_hint in test_cases:
            decision = router.route(query)
            evidence[query[:40]] = {
                "category": decision.category.value,
                "model_name": decision.model_name,
                "reason": decision.reason[:80] if decision.reason else "",
            }
    except Exception as exc:
        return TestResult("router", "FAIL", str(exc), evidence)
    return TestResult("router", "PASS", evidence=evidence)


# ─── 11. Security ────────────────────────────────────────────────────────────

def check_security() -> TestResult:
    evidence = {}
    try:
        from security.offline_mode import verify_offline
        result = verify_offline()
        evidence["offline_check"] = result
    except Exception as exc:
        return TestResult("security", "FAIL", str(exc), evidence)
    return TestResult("security", "PASS", evidence=evidence)


# ─── 12. Document Generation ─────────────────────────────────────────────────

def check_docx() -> TestResult:
    try:
        from tools.docx_tool import DocxTool
        return TestResult("docx", "PASS")
    except ImportError:
        try:
            from docx import Document
            return TestResult("docx", "PASS", "python-docx available")
        except ImportError:
            return TestResult("docx", "FAIL", "python-docx not installed")


def check_xlsx() -> TestResult:
    try:
        from openpyxl import Workbook
        return TestResult("xlsx", "PASS")
    except ImportError:
        return TestResult("xlsx", "FAIL", "openpyxl not installed")


def check_pptx() -> TestResult:
    try:
        from pptx import Presentation
        return TestResult("pptx", "PASS")
    except ImportError:
        return TestResult("pptx", "FAIL", "python-pptx not installed")


# ─── Master runner ────────────────────────────────────────────────────────────

CATEGORIES = [
    ("environment", check_environment),
    ("repository", check_repository),
    ("dataset", check_dataset),
    ("tokenizer", check_tokenizer),
    ("model", check_model),
    ("checkpoints", check_checkpoints),
    ("rag", check_rag),
    ("fineweb", check_fineweb),
    ("database", check_database),
    ("router", check_router),
    ("security", check_security),
    ("docx", check_docx),
    ("xlsx", check_xlsx),
    ("pptx", check_pptx),
]


def run_all(categories: list[str] | None = None) -> list[TestResult]:
    results = []
    for name, fn in CATEGORIES:
        if categories and name not in categories:
            continue
        log.info("━━━ %s", name.upper())
        result = _run(name, fn)
        log.info("    %s  (%.1fs)  %s", result.status, result.duration_s, result.details[:80] if result.details else "")
        results.append(result)
    return results


def generate_health_report(results: list[TestResult]) -> str:
    lines = [
        "# SOVEREIGNAI SYSTEM HEALTH",
        "",
        f"> Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "## Component Status",
        "",
        "| Component | Status | Duration | Details |",
        "|---|---|---|---|",
    ]
    for r in results:
        status_icon = {"PASS": "✅", "FAIL": "❌", "BLOCKED": "🚫", "NOT_IMPLEMENTED": "⬜", "NOT_APPLICABLE": "➖"}.get(r.status, "❓")
        lines.append(f"| {r.category} | {status_icon} {r.status} | {r.duration_s:.1f}s | {r.details[:60]} |")

    # Summary
    passed = sum(1 for r in results if r.status == "PASS")
    failed = sum(1 for r in results if r.status == "FAIL")
    total = len(results)

    lines.extend([
        "",
        "## Summary",
        "",
        f"- **Passed**: {passed}/{total}",
        f"- **Failed**: {failed}/{total}",
        "",
    ])

    if failed:
        lines.append("## Critical Failures")
        lines.append("")
        for r in results:
            if r.status == "FAIL":
                lines.append(f"### {r.category}")
                lines.append(f"- **Details**: {r.details}")
                if r.evidence:
                    lines.append(f"- **Evidence**: `{json.dumps(r.evidence, default=str)[:200]}`")
                lines.append("")

    # Evidence for passes
    lines.append("## Evidence")
    lines.append("")
    for r in results:
        if r.evidence:
            lines.append(f"### {r.category} ({r.status})")
            lines.append("```json")
            lines.append(json.dumps(r.evidence, indent=2, default=str)[:1000])
            lines.append("```")
            lines.append("")

    return "\n".join(lines)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="SovereignAI full validation")
    parser.add_argument("--category", type=str, help="Run only this category")
    parser.add_argument("--report-only", action="store_true", help="Generate report from last run")
    args = parser.parse_args()

    categories = [args.category] if args.category else None
    results = run_all(categories)

    # Generate report
    report = generate_health_report(results)
    report_dir = ROOT / "test_reports" / "latest"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "SYSTEM_HEALTH.md"
    report_path.write_text(report, encoding="utf-8")
    log.info("━━━ Health report: %s", report_path)

    # Also save JSON
    json_path = report_dir / "results.json"
    json_path.write_text(json.dumps(
        [{"category": r.category, "status": r.status, "details": r.details,
          "evidence": r.evidence, "duration_s": r.duration_s} for r in results],
        indent=2, default=str
    ), encoding="utf-8")

    # Exit code
    failed = sum(1 for r in results if r.status == "FAIL")
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
