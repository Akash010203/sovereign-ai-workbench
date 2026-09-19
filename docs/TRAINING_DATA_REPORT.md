# SovereignAI — Training Data Report

> Generated: 2026-09-19 by forensic audit.

## Blended Training Corpus (~1.6 Billion Tokens)

### File: `data/processed/blended_train.txt`

| Metric | Value |
|---|---|
| File size | 3.29 GB (3,288,024,343 bytes) |
| Estimated tokens | ~1.6 billion (at ~2 bytes/token) |
| Format | Plain text, newline-separated |
| Git tracked | No (gitignored) |
| Reproducible | Yes — via `scripts/prepare_blended_corpus.py` |

### Composition (4-Source Blend)

| Source | Dataset ID | License | Contribution |
|---|---|---|---|
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | ODC-By 1.0 | Primary web knowledge |
| OpenOrca | `Open-Orca/OpenOrca` | Apache 2.0 | Instruction-following ability |
| Nemotron | NVIDIA Nemotron | permissive | Conversational quality |
| Maintenance | Domain-specific maintenance dataset | varies | Industrial/engineering domain |

### Validation Split

| File | Size | Purpose |
|---|---|---|
| `blended_val.txt` | 66 MB | Held-out validation for loss computation |

### Build Pipeline

```
scripts/download_blend_sources.py    ← Download from HuggingFace (one-time)
         ↓
scripts/prepare_blended_corpus.py    ← Clean, filter, deduplicate, blend
         ↓
data/processed/blended_train.txt     ← 3.29 GB training text
data/processed/blended_val.txt       ← 66 MB validation text
```

### Hardware Constraints

| Constraint | Value |
|---|---|
| GPU | RTX 4050 (6 GB VRAM) |
| Model | Industrial 80M (~83M params) |
| Max batch size | micro-batch 1, gradient accumulation 16 |
| Context length | 256 tokens |
| AMP | Yes (mixed precision) |

**Note**: Training 1.6B tokens from scratch on a 6 GB GPU is impractical in a single run. The architecture supports:
- Configurable dataset subset sizes
- Checkpointing and resume
- Streaming data loading
- Gradient accumulation for effective larger batches

### Training completed: 20,000 steps with loss convergence (9.82 → 2.37)

## OpenOrca Data

| File | Size | Format | Purpose |
|---|---|---|---|
| `openorca_instruct.jsonl` | 416 MB | JSONL | Instruction tuning pairs |
| `openorca_pretrain.txt` | 490 MB | Text | Pretraining text from OpenOrca |
| `openorca_val.txt` | 9.9 MB | Text | OpenOrca validation |

## Demo Corpus (Small, Project-Authored)

| File | Size | Purpose |
|---|---|---|
| `data/raw/synthetic_demo_corpus.txt` | 4.7 KB | Original synthetic industrial text |
| `data/processed/train.txt` | 3.9 KB | Demo training split |
| `data/processed/val.txt` | 850 B | Demo validation split |

These are git-tracked, project-authored files used for smoke testing.

## FineWeb-Edu — Dual-Purpose Usage

FineWeb-Edu is correctly used in TWO separate pipelines:

1. **Training**: Blended into `blended_train.txt` for LLM weight training
2. **RAG**: Chunked and embedded in `data/knowledge/fineweb_edu_1p6b_embed/` for retrieval

These are separate processes with separate outputs. The training pipeline produces token IDs; the RAG pipeline produces embedding vectors. They are NOT interchangeable.

## Data Integrity

All training data files are gitignored but reproducible:
- Raw sources cached in `data/raw/hf_sources/` (downloaded from HuggingFace)
- Processing scripts in `scripts/` (git-tracked)
- Processing is deterministic — same inputs produce same outputs
