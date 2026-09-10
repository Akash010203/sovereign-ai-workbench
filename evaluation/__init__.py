"""
evaluation/__init__.py — Evaluation metrics for SovereignAI.

PHASE 22 — Model and System Evaluation

This package provides:
  - Perplexity evaluation on the validation set
  - BLEU and ROUGE scores for text generation quality
  - Routing accuracy on labeled test queries
  - Retrieval precision@K for RAG evaluation
  - System throughput (tokens/sec) and latency measurements
  - An aggregate evaluation report for the SIH demo
"""
from evaluation.metrics import (
    compute_perplexity,
    compute_bleu,
    compute_rouge_1,
    RoutingEvaluator,
    RetrievalEvaluator,
    LatencyBenchmark,
    run_full_evaluation,
)

__all__ = [
    "compute_perplexity",
    "compute_bleu",
    "compute_rouge_1",
    "RoutingEvaluator",
    "RetrievalEvaluator",
    "LatencyBenchmark",
    "run_full_evaluation",
]
