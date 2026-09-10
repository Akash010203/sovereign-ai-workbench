"""
scripts/evaluate.py — Run the Phase 22 evaluation suite.

Usage:
    python scripts\\evaluate.py                         # routing + metrics only
    python scripts\\evaluate.py --checkpoint checkpoints/exp2_step00300_final.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.logging_setup import setup_logging
from evaluation.metrics import run_full_evaluation


def parse_args():
    p = argparse.ArgumentParser(description="SovereignAI Evaluation Suite")
    p.add_argument("--checkpoint", default="", help="MiniLLM checkpoint for perplexity/latency")
    p.add_argument("--rag-index",  default="data/rag_index.json", help="RAG index JSON")
    p.add_argument("--output",     default="logs/evaluation_report.json")
    return p.parse_args()


def main():
    setup_logging("evaluate")
    args   = parse_args()
    model  = None
    tok    = None
    retriever = None

    # Load model if checkpoint given
    if args.checkpoint and Path(args.checkpoint).exists():
        import torch
        from models.custom_minilm.checkpoint import load_model_from_checkpoint
        from tokenizer.tokenizer import ByteLevelBPETokenizer
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, meta = load_model_from_checkpoint(args.checkpoint, device=device)
        tok_path    = meta.get("tokenizer_path", "tokenizer/vocab/demo_bpe_vocab.json")
        tok = ByteLevelBPETokenizer.load(ROOT / tok_path)
        print(f"Loaded model: {meta['model_params']:,} params  step={meta['step']}")

    # Load RAG index if available
    index_path = ROOT / args.rag_index
    if index_path.exists():
        from rag.index import LocalVectorIndex
        from rag.retriever import Retriever
        from rag.embeddings import get_embedder
        idx = LocalVectorIndex()
        idx.load(index_path)
        retriever = Retriever(idx)
        print(f"RAG index: {len(idx)} chunks")

    # Run evaluation
    report = run_full_evaluation(
        model=model, tokenizer=tok, retriever=retriever,
        output_path=args.output,
    )

    # Print summary
    print("\n" + "=" * 60)
    print("EVALUATION REPORT — SovereignAI")
    print("=" * 60)

    if "routing" in report:
        r = report["routing"]
        print(f"\n  Task Routing Accuracy : {r['accuracy']*100:.1f}%  ({r['correct']}/{r['total']})")
        if r["errors"]:
            print("  Routing errors:")
            for e in r["errors"][:5]:
                print(f"    '{e['text'][:50]}' -> got '{e['predicted']}', want '{e['expected']}'")

    if "retrieval" in report:
        r = report["retrieval"]
        print(f"  RAG Precision@{r['k']}     : {r['precision_at_k']*100:.1f}%  ({r['hits']}/{r['total']})")

    if "perplexity" in report:
        print(f"  MiniLLM Perplexity    : {report['perplexity']['value']:.2f}")

    if "latency" in report:
        r = report["latency"]
        print(f"  Generation Latency    : {r['avg_latency_s']:.3f}s avg  ({r['tokens_per_sec']:.1f} tok/s)  [{r['device']}]")

    if "bleu_sample" in report:
        print(f"  BLEU (sample)         : {report['bleu_sample']['score']:.4f}")

    if "rouge1_sample" in report:
        print(f"  ROUGE-1 F1 (sample)   : {report['rouge1_sample']['f1']:.4f}")

    print(f"\n  Full report: {args.output}")
    print("=" * 60)


if __name__ == "__main__":
    main()
