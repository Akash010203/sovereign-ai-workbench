# SovereignAI — Recovery Timeline

> Generated: 2026-09-19 by forensic audit.

## Commit History (Oldest → Newest)

| # | Date | Commit | State | What Worked | What Changed | Evidence |
|---|---|---|---|---|---|---|
| 1 | Sep 10 21:02 | `2f00bec` | INITIAL | Project skeleton, SIH 2026 Phase 0-24 structure | N/A — first commit | Initial commit message |
| 2 | Sep 10 21:07 | `08fd084` | BUILDING | Demo corpus, BPE tokenizer training | Added `data/demo/`, `data/metadata/`, train/val splits, BPE vocab | Commit diff |
| 3 | Sep 10 23:02 | `2abd7e7` | BUILDING | Training pipeline, safety tests | Added training loop, audit docs | Commit message |
| 4 | Sep 11 12:18 | `fc555b5` | FUNCTIONAL | Complete workbench: custom LLM arch, offline verify, 8K vocab, exhaustive docs | Major feature addition — all core components | Commit message + docs |
| 5 | Sep 12 13:27 | `98466f9` | FUNCTIONAL | Fine-tuned checkpoint, RAG citations, offline mode | Generation fixes, citation format | Commit message |
| 6 | Sep 17 10:32 | `db1c694` | **BEST STATE** | Industrial 80M model, blended dataset, RAG agent integration | Major model upgrade, data pipeline | Commit message + training logs |
| 7 | Sep 17 22:54 | `85a7765` | **BEST STATE** | Hybrid FineWeb RAG, 4-source corpus blend, ingestion scripts | Added FineWeb index, hybrid retriever, corpus blending | Commit message |
| 8 | Sep 18 17:02 | `a50fc4e` | MERGED | Merge commit | Branch merge | Git log |
| 9 | Sep 18 17:51 | `3d5fdd8` | HEAD | Documentation enhancement (Mermaid maps, runtime traces) | Docs only — code unchanged | Commit message |
| 10 | UNCOMMITTED | Working tree | **REGRESSED** | Backend rewrite, RAG embedder stripped, Ollama adapter deleted | 41 files modified, 1 deleted | `git diff --stat HEAD` |

## Critical State Transitions

### LAST KNOWN GOOD STATE: Commit `85a7765` (Sep 17)
- Hybrid FineWeb RAG operational with SentenceTransformer embeddings
- Industrial 80M model trained (20,000 steps)
- 4-source blended corpus prepared
- Full ingestion pipeline with PDF, DOCX, XLSX, PPTX support
- 10,975 RAG chunks indexed with 384-dim embeddings

### FIRST BAD STATE: Uncommitted working-tree changes
- `rag/embeddings.py`: SentenceTransformerEmbedder removed, TF-IDF only
- `models/adapters/openweight_adapter.py`: DELETED (Ollama integration)
- `models/registry.py`: Stripped to MiniLLM-only registry
- `app/backend/app.py`: Major rewrite — user_knowledge separation, extractive replies
- Multiple test files modified

### What Disappeared
1. **SentenceTransformerEmbedder class** — removed from `rag/embeddings.py`
2. **OpenWeight (Ollama) adapter** — file deleted entirely
3. **Dual-embedder fallback logic** — removed from `get_embedder()`
4. **FineWeb 1.6B index compatibility** — broken by dimension mismatch (384 → 512)

### What Was NOT Lost (Data Intact)
1. All checkpoint files (102 GB) — gitignored but present
2. All RAG indexes — `rag_index.json` (384-dim), `rag_index_tfidf.json` (512-dim), `user_knowledge_index.json` (512-dim)
3. FineWeb 1.6B knowledge shards (17.1 GB) — chunks + embeddings intact
4. Training data — `blended_train.txt` (3.29 GB), OpenOrca, etc.
5. SQLite database — `sovereign_ai.db` (176 KB)
6. Training logs — all experiment logs present
7. Tokenizer vocabularies — all 3 vocabs intact

## Recovery Possibility

| Component | Recovery Method | Effort |
|---|---|---|
| SentenceTransformer embedder | Restore from `git show HEAD:rag/embeddings.py` | Trivial |
| OpenWeight adapter | Restore from `git show HEAD:models/adapters/openweight_adapter.py` | Trivial |
| RAG index (384-dim) | Already exists: `data/rag_index.json` | None |
| FineWeb index (384-dim) | Already exists: `data/knowledge/fineweb_edu_1p6b_embed/` | None |
| TF-IDF RAG index | Already exists: `data/rag_index_tfidf.json` | None |
| Checkpoints | Already exist in `checkpoints/` | None |
| Training data | Already exists in `data/processed/` | None |

## Stash Contents

- `stash@{0}`: "On main: !!GitHub_Desktop\<main\>" — autosaved by GitHub Desktop, needs inspection
