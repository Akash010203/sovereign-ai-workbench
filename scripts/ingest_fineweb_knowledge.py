"""Stream FineWeb-Edu into resumable RAG knowledge chunks and embeddings.

Examples:
  # Small, safe validation run (chunks only; no model download)
  python scripts/ingest_fineweb_knowledge.py --target-tokens 100000 --max-source-rows 1000

  # Full target, with a locally cached sentence-transformer embedding model
  python scripts/ingest_fineweb_knowledge.py --target-tokens 1600000000 --embed --require-neural
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rag.embeddings import SentenceTransformerEmbedder, get_embedder
from rag.fineweb import FineWebChunkConfig, FineWebKnowledgeIngestor, stream_fineweb_edu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream FineWeb-Edu into sharded local knowledge chunks")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data" / "knowledge" / "fineweb_edu")
    parser.add_argument("--target-tokens", type=int, default=1_600_000_000,
                        help="Estimated normalized-text token budget (default: 1.6B).")
    parser.add_argument("--chunk-tokens", type=int, default=384,
                        help="Estimated tokens per retrieval chunk (default: 384).")
    parser.add_argument("--overlap-tokens", type=int, default=48)
    parser.add_argument("--chars-per-token", type=float, default=4.0,
                        help="Transparent token estimator; use training-tokenizer measurement for an exact training budget.")
    parser.add_argument("--min-document-chars", type=int, default=200)
    parser.add_argument("--shard-size", type=int, default=50_000)
    parser.add_argument("--embedding-batch-size", type=int, default=64)
    parser.add_argument("--max-source-rows", type=int, default=None,
                        help="Bound source rows for a smoke test; omit for the configured token target.")
    parser.add_argument("--no-resume", action="store_true", help="Start a new run in an empty output directory.")
    parser.add_argument("--embed", action="store_true", help="Write float32 embedding shards beside chunk JSONL files.")
    parser.add_argument("--require-neural", action="store_true",
                        help="Fail instead of silently using the hashing TF-IDF fallback.")
    parser.add_argument("--network-retries", type=int, default=20,
                        help="Automatic source-stream retries after transient Hugging Face network failures (default: 20).")
    parser.add_argument("--retry-delay-seconds", type=int, default=15,
                        help="Wait before reopening a failed source stream (default: 15).")
    return parser.parse_args()


def _is_transient_network_error(error: BaseException) -> bool:
    """Recognise network errors without retrying embedding/data-format bugs."""
    if isinstance(error, (ConnectionError, TimeoutError, OSError)):
        return True
    message = str(error).lower()
    return any(marker in message for marker in (
        "timed out", "getaddrinfo", "client has been closed", "connection reset",
        "connection aborted", "temporary failure", "network is unreachable",
    ))


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    if args.require_neural and not args.embed:
        raise SystemExit("--require-neural only applies when --embed is also set.")
    if args.network_retries < 0 or args.retry_delay_seconds < 0:
        raise SystemExit("--network-retries and --retry-delay-seconds cannot be negative.")
    if args.no_resume and args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise SystemExit("--no-resume requires an empty --output-dir to avoid mixing two runs.")

    config = FineWebChunkConfig(
        target_tokens=args.target_tokens,
        chunk_tokens=args.chunk_tokens,
        overlap_tokens=args.overlap_tokens,
        chars_per_token=args.chars_per_token,
        min_document_chars=args.min_document_chars,
        shard_size=args.shard_size,
    )
    embedder = get_embedder() if args.embed else None
    if args.require_neural and embedder is not None and not isinstance(embedder, SentenceTransformerEmbedder):
        raise SystemExit("A locally cached sentence-transformer is required; refusing the non-semantic TF-IDF fallback.")

    ingestor = FineWebKnowledgeIngestor(
        args.output_dir, config, embedder=embedder, embedding_batch_size=args.embedding_batch_size,
    )
    for attempt in range(args.network_retries + 1):
        try:
            stats = ingestor.ingest(
                stream_fineweb_edu(config_name=config.dataset_config),
                resume=not args.no_resume,
                max_source_rows=args.max_source_rows,
            )
            break
        except Exception as exc:
            if not _is_transient_network_error(exc) or attempt >= args.network_retries:
                raise
            logging.warning(
                "FineWeb source stream failed (%s). A checkpoint was saved; retrying %d/%d in %ds.",
                exc, attempt + 1, args.network_retries, args.retry_delay_seconds,
            )
            time.sleep(args.retry_delay_seconds)
    print(json.dumps({"output_dir": str(args.output_dir), **stats.__dict__}, indent=2))


if __name__ == "__main__":
    main()
