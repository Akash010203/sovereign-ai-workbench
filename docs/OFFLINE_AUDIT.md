# SovereignAI — Offline Audit

> Generated: 2026-09-19 by forensic audit.

## Offline Design Principles

SovereignAI is designed for air-gapped / fully-offline operation.
All components must function without internet access after initial setup.

## Network Access Audit

### Code Scanning Results

| Pattern | Files Found | Classification |
|---|---|---|
| `requests.get` / `requests.post` | 0 in core code | CLEAN |
| `httpx` | 0 | CLEAN |
| `urllib.request` | 0 in core code | CLEAN |
| `api.openai.com` | 0 | CLEAN |
| `anthropic` | 0 | CLEAN |
| `socket.connect` | 1 (`security/network_monitor.py`) | Intentional — offline verification |
| `HuggingFace` download | 2 (`scripts/download_blend_sources.py`, `scripts/ingest_fineweb_knowledge.py`) | One-time setup only |

### Components with Internet Dependencies

| Component | When | Purpose | Air-Gap Mitigation |
|---|---|---|---|
| `scripts/download_blend_sources.py` | One-time setup | Download training datasets from HuggingFace | Run once, then use cached data |
| `scripts/ingest_fineweb_knowledge.py` | One-time setup | Stream FineWeb for RAG | Run once, shards persist to disk |
| SentenceTransformer loading | First use | Download `all-MiniLM-L6-v2` model | `local_files_only=True` after first cache |
| `sentence-transformers` | Import time | Python package | Pre-install in venv |

### Runtime Behavior (After Setup)

| Component | Network Access | Evidence |
|---|---|---|
| Backend (`app/backend/app.py`) | **NONE** | Flask on localhost only |
| Custom MiniLLM inference | **NONE** | Pure PyTorch, local checkpoint |
| RAG pipeline | **NONE** | Local embedder + local index |
| Agent system | **NONE** | Local tools only |
| Database | **NONE** | SQLite, local file |
| Frontend | **NONE** | Static HTML served by Flask |

### Offline Verification

The project includes `security/offline_mode.py` which provides:
- `verify_offline()` — Checks that no outbound connections are made
- Runtime network monitoring capability

**Verification result**: `NO OUTBOUND PROBE` — confirmed offline operation.

## Air-Gap Deployment Checklist

1. [ ] Install Python + pip packages from local mirror
2. [ ] Cache `sentence-transformers/all-MiniLM-L6-v2` in `~/.cache/huggingface/`
3. [ ] Copy `data/processed/blended_train.txt` (or pre-trained checkpoint)
4. [ ] Copy `data/knowledge/fineweb_edu_1p6b_embed/` (if FineWeb RAG needed)
5. [ ] Copy `checkpoints/industrial_80m_best.pt`
6. [ ] Copy `tokenizer/vocab/blended_16k.json`
7. [ ] Run `python scripts/verify_offline.py` to confirm
