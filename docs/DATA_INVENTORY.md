# SovereignAI — Data Inventory

> Generated: 2026-09-19 by forensic audit.

## Classification Key

| Code | Meaning |
|---|---|
| TRAINING | Used to train the custom LLM weights |
| RAG | Ingested into the retrieval-augmented generation knowledge base |
| APPLICATION | Runtime state (SQLite, sessions, config) |
| CACHE | Intermediate/generated files |
| CHECKPOINT | Model weight snapshots |
| INDEX | Vector/search index files |

---

## Training Data

| Name | Location | Format | Size | Approx Records | Purpose | Source | License | Git Tracked? |
|---|---|---|---|---|---|---|---|---|
| blended_train.txt | `data/processed/` | Plain text | 3.29 GB | ~1.6B tokens | Primary training corpus (4-source blend) | FineWeb-Edu + OpenOrca + Nemotron + Maintenance | Mixed (Apache 2.0 / CC-BY) | No (gitignored) |
| blended_val.txt | `data/processed/` | Plain text | 66 MB | ~32M tokens est. | Validation split for blended corpus | Same as above | Same | No |
| openorca_instruct.jsonl | `data/processed/` | JSONL | 416 MB | ~90K+ rows | Instruction fine-tuning dataset | Open-Orca (HuggingFace) | Apache 2.0 | No |
| openorca_pretrain.txt | `data/processed/` | Plain text | 490 MB | ~240M tokens est. | OpenOrca in pretrain text format | Open-Orca | Apache 2.0 | No |
| openorca_val.txt | `data/processed/` | Plain text | 9.9 MB | ~4.8M tokens est. | OpenOrca validation split | Open-Orca | Apache 2.0 | No |
| train.txt | `data/processed/` | Plain text | 3.9 KB | ~1.9K tokens | Demo training split (synthetic) | Project-authored | Project license | Yes |
| val.txt | `data/processed/` | Plain text | 850 B | ~425 tokens | Demo validation split (synthetic) | Project-authored | Project license | Yes |
| synthetic_demo_corpus.txt | `data/raw/` | Plain text | 4.7 KB | ~2.3K tokens | Source for demo train/val splits | Project-authored | Project license | Yes |

### HuggingFace Source Cache

| Name | Location | Purpose | Git Tracked? |
|---|---|---|---|
| hf_sources/ | `data/raw/hf_sources/` | Downloaded HF dataset cache | No (gitignored) |
| fineweb_edu/ | `data/raw/hf_sources/fineweb_edu/` | FineWeb-Edu raw downloads | No |
| maintenance/ | `data/raw/hf_sources/maintenance/` | Maintenance dataset downloads | No |
| nemotron/ | `data/raw/hf_sources/nemotron/` | Nemotron dataset downloads | No |
| openorca/ | `data/raw/hf_sources/openorca/` | OpenOrca dataset downloads | No |

---

## RAG Knowledge Data

| Name | Location | Format | Size | Records | Embed Dim | Purpose | Status |
|---|---|---|---|---|---|---|---|
| rag_index.json | `data/` | JSON | 136 MB | 10,975 | 384 | Primary RAG index (SentenceTransformer) | STALE — dim mismatch with current code |
| rag_index_tfidf.json | `data/` | JSON | 39 MB | 10,975 | 512 | TF-IDF compatible RAG index | COMPATIBLE with current code |
| user_knowledge_index.json | `data/` | JSON | 90 MB | 10,975 | 512 | User knowledge base (loaded by backend) | ACTIVE |

### RAG Source Documents

| Source | Records in Index | Location |
|---|---|---|
| maintenance_parquet_row* | 10,000 | Ingested from HF maintenance dataset |
| compressor_seal_replacement_sop.txt | varies | `data/demo/` |
| equipment_inspection_report.txt | varies | `data/demo/` |
| maintenance_log_scan.txt | varies | `data/demo/` |
| sovereignai_sih_demo_maintenance_knowledge_pack.pdf | varies | Generated/provided |
| synthetic_demo_corpus.txt | varies | `data/raw/` |
| README.md | varies | Project README |

### FineWeb 1.6B Knowledge Shards (RAG, NOT Training)

| Name | Location | Format | Size | Chunks | Embed Dim | Purpose |
|---|---|---|---|---|---|---|
| chunks-00000..00104.jsonl | `data/knowledge/fineweb_edu_1p6b_embed/` | JSONL | ~98 MB each | 5,240,427 total | N/A | Text chunks for retrieval |
| embeddings-00000..00104.f32 | `data/knowledge/fineweb_edu_1p6b_embed/` | Float32 binary | ~77 MB each | 5,240,427 total | 384 | SentenceTransformer embeddings |
| fineweb_edu_manifest.json | `data/knowledge/fineweb_edu_1p6b_embed/` | JSON | 1 KB | N/A | N/A | Ingestion metadata |

**Total FineWeb knowledge store: 17.1 GB** (105 chunk shards + 105 embedding shards)

**CRITICAL**: The manifest explicitly states `"purpose": "RAG knowledge chunks; this does not train or modify the language model weights."` This is correctly classified as RAG data.

### FineWeb Small Test Set

| Name | Location | Format | Size | Chunks |
|---|---|---|---|---|
| chunks-00000.jsonl | `data/knowledge/fineweb_edu/` | JSONL | 616 KB | 336 |
| fineweb_edu_manifest.json | `data/knowledge/fineweb_edu/` | JSON | 998 B | N/A |

---

## Checkpoints

| Series | Location | Count | Size Each | Total | Architecture |
|---|---|---|---|---|---|
| exp1_smoke_* | `checkpoints/` | 5 | ~56 MB | ~280 MB | Small (~5M params) |
| exp4_openorca_* | `checkpoints/` | ~280 | 56-207 MB | ~50 GB | Medium (~12M) / varied |
| industrial_80m_* | `checkpoints/` | ~42 | ~951 MB | ~40 GB | Industrial 80M (~83M params) |
| best.pt | `checkpoints/` | 1 | 207 MB | 207 MB | Best OpenOrca checkpoint |
| industrial_80m_best.pt | `checkpoints/` | 1 | 951 MB | 951 MB | Best industrial checkpoint |
| industrial_80m_smoke.pt | `checkpoints/` | 1 | 951 MB | 951 MB | Industrial smoke test |
| finetuned/ | `checkpoints/finetuned/` | dir | unknown | unknown | Fine-tuned variants |
| _audit_pretrain.pt | `checkpoints/` | 1 | 56 MB | 56 MB | Audit checkpoint |

**Total checkpoint storage: ~102 GB**

---

## Application State

| Name | Location | Format | Size | Purpose | Git Tracked? |
|---|---|---|---|---|---|
| sovereign_ai.db | `data/` | SQLite | 176 KB | Conversations, tasks | No |
| audit.log | `logs/` | Text | varies | Security audit log | No |
| sovereign_ai.log | `logs/` | Text | varies | Application log | No |
| backend.out.log | `logs/` | Text | varies | Backend stdout | No |
| backend.err.log | `logs/` | Text | varies | Backend stderr | No |
| offline_proof.log | `logs/` | Text | varies | Offline verification log | No |

---

## Tokenizers

| Name | Location | Format | Size | Vocab Size | Purpose | Git Tracked? |
|---|---|---|---|---|---|---|
| demo_bpe_vocab.json | `tokenizer/vocab/` | JSON | 65 KB | ~600 | Demo/smoke-test tokenizer | Yes |
| bpe_8k.json | `tokenizer/vocab/` | JSON | 1.6 MB | ~8,000 | Medium tokenizer | Yes |
| blended_16k.json | `tokenizer/vocab/` | JSON | 3.6 MB | ~16,000 | Industrial tokenizer (used by 80M model) | Yes |

---

## Training Logs

| Name | Location | Format | Size | Steps | Final Loss |
|---|---|---|---|---|---|
| exp1_smoke_log.json | `logs/` | JSON | varies | ~50 | N/A |
| exp4_openorca_log.json | `logs/` | JSON | varies | 30,000 | N/A |
| industrial_80m_4050_log.json | `logs/` | JSON | varies | 20,000 | train=2.37, val=3.57 |
