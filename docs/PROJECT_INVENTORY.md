# SovereignAI — Project Inventory

> Generated: 2026-09-19 by forensic audit.

## Root Directory Structure

| Path | Type | Purpose | Git Tracked | Criticality |
|---|---|---|---|---|
| `agents/` | Directory | Multi-step agent system | Yes | HIGH |
| `app/` | Directory | Frontend + Flask backend | Yes | HIGH |
| `checkpoints/` | Directory | Model weight snapshots (~117 GB) | No (gitignored) | CRITICAL |
| `core/` | Directory | Configuration, logging | Yes | HIGH |
| `data/` | Directory | Datasets, indexes, database | Partial | CRITICAL |
| `database/` | Directory | SQLite ORM layer | Yes | MEDIUM |
| `demo/` | Directory | Demo scenarios and runner | Yes | LOW |
| `docs/` | Directory | Documentation | Yes | MEDIUM |
| `evaluation/` | Directory | Model evaluation scripts | Yes | MEDIUM |
| `logs/` | Directory | Runtime logs | No (gitignored) | LOW |
| `models/` | Directory | Custom MiniLLM + adapters | Yes | CRITICAL |
| `output/` | Directory | Generated documents | No (gitignored) | LOW |
| `rag/` | Directory | RAG pipeline (ingest, embed, index, retrieve) | Yes | CRITICAL |
| `router/` | Directory | Task classification/routing | Yes | HIGH |
| `scripts/` | Directory | Build, train, ingest, validate scripts | Yes | HIGH |
| `security/` | Directory | Offline mode, network monitor, audit | Yes | HIGH |
| `test_reports/` | Directory | Validation reports | No | LOW |
| `tests/` | Directory | Test suites | Partial | HIGH |
| `tmp/` | Directory | Temporary files | No | LOW |
| `tokenizer/` | Directory | BPE tokenizer | Yes | CRITICAL |
| `tools/` | Directory | Agent tools (sandbox, docx, xlsx, pptx) | Yes | HIGH |
| `training/` | Directory | Training loop, dataset, scheduler | Yes | CRITICAL |

## Source Code Files

### agents/
| File | Size | Purpose | Referenced By | Produces |
|---|---|---|---|---|
| `agent.py` | ~13 KB | Main agent orchestrator | `app/backend/app.py` | Task execution |
| `executor.py` | varies | Tool execution engine | `agent.py` | Tool results |
| `state.py` | varies | Task state management | `agent.py` | TaskState objects |
| `verifier.py` | varies | Result verification | `agent.py` | Verification results |

### app/
| File | Size | Purpose | Referenced By | Produces |
|---|---|---|---|---|
| `backend/app.py` | ~21 KB | Flask REST API server | Entry point | HTTP responses |
| `frontend/index.html` | varies | Single-page web UI | Backend static serving | User interface |

### models/custom_minilm/
| File | Size | Purpose | Referenced By | Produces |
|---|---|---|---|---|
| `model.py` | ~15 KB | Full MiniLLM Transformer | Adapters, training | Model forward pass |
| `config.py` | ~4 KB | Hyperparameter configs | `model.py`, training | Config objects |
| `attention.py` | ~4.5 KB | Causal multi-head attention + RoPE | `block.py` | Attention output |
| `block.py` | ~3.3 KB | Transformer block (norm+attn+FFN) | `model.py` | Block output |
| `embeddings.py` | ~1.8 KB | Token embedding layer | `model.py` | Embedded tokens |
| `ffn.py` | ~4.4 KB | SwiGLU feed-forward network | `block.py` | FFN output |
| `normalization.py` | ~1.8 KB | RMSNorm layer | `block.py`, `model.py` | Normalized tensors |
| `rotary.py` | ~3.2 KB | Rotary positional encoding (RoPE) | `attention.py` | Position-encoded Q/K |
| `generate.py` | ~5 KB | Text generation (sampling) | Adapters | Generated text |
| `checkpoint.py` | ~6.6 KB | Save/load model checkpoints | Training, adapters | Checkpoint files |

### rag/
| File | Size | Purpose | Referenced By | Produces |
|---|---|---|---|---|
| `embeddings.py` | ~5 KB | SentenceTransformer + TF-IDF embedder | Ingest, retriever | Embedding vectors |
| `chunking.py` | ~3.5 KB | Document chunking (sentence/fixed) | `ingest.py` | TextChunk objects |
| `ingest.py` | ~4 KB | Document ingestion pipeline | Backend, scripts | Indexed chunks |
| `index.py` | ~5.3 KB | LocalVectorIndex (in-memory flat) | Retriever, ingest | Search results |
| `retriever.py` | ~2.9 KB | Query → ranked results | Backend, tools | RetrievalResult |
| `citations.py` | ~1.4 KB | Citation formatting | Backend | Citation strings |
| `hybrid.py` | ~1.3 KB | Hybrid retriever (local+FineWeb+lexical) | Backend | Merged results |
| `fineweb.py` | ~16 KB | FineWeb ingestion pipeline | Scripts | Chunk/embedding shards |
| `fineweb_index.py` | ~6.8 KB | Disk-backed FineWeb search | `hybrid.py` | FineWeb results |
| `fineweb_lexical.py` | ~3.5 KB | FineWeb lexical/keyword search | `hybrid.py` | Lexical results |

### training/
| File | Size | Purpose | Referenced By | Produces |
|---|---|---|---|---|
| `trainer.py` | ~16.5 KB | Main training loop | Scripts | Checkpoints, logs |
| `dataset.py` | ~6.5 KB | Training dataset + dataloader | `trainer.py` | Training batches |
| `scheduler.py` | ~3.1 KB | Learning rate scheduler | `trainer.py` | LR schedule |
| `evaluation.py` | ~2.9 KB | Model evaluation metrics | `trainer.py` | Eval results |

### tokenizer/
| File | Size | Purpose | Referenced By | Produces |
|---|---|---|---|---|
| `tokenizer.py` | ~9.7 KB | BPE tokenizer implementation | Training, adapters | Token IDs |
| `trainer.py` | ~5.2 KB | BPE vocabulary training | Scripts | Vocab files |

### scripts/
| File | Size | Purpose |
|---|---|---|
| `train_model.py` | ~12.8 KB | Model training entry point |
| `prepare_blended_corpus.py` | ~18.4 KB | 4-source corpus blending |
| `prepare_openorca.py` | ~11.7 KB | OpenOrca data preparation |
| `ingest_rag.py` | ~8.7 KB | RAG knowledge ingestion |
| `ingest_fineweb_knowledge.py` | ~5.6 KB | FineWeb knowledge ingestion |
| `finetune_instruct.py` | ~20.7 KB | Instruction fine-tuning |
| `download_blend_sources.py` | ~5.1 KB | Dataset download script |
| `train_tokenizer.py` | ~3.8 KB | Tokenizer training |
| `train_pipeline.py` | ~10.6 KB | Full training pipeline |
| `run_full_validation.py` | ~8 KB | Master validation runner |
| `create_recovery_snapshot.py` | ~5 KB | System state snapshot |
| `run_inference.py` | ~2.8 KB | Inference testing |
| `verify_offline.py` | ~1.8 KB | Offline verification |
| `verify_gpu.py` | ~1.7 KB | GPU verification |
| `start_backend.py` | ~775 B | Backend launcher |
| `run_demo.py` | ~425 B | Demo launcher |

## Test Suite

| File | Tests | Coverage |
|---|---|---|
| `tests/test_rag_regression.py` | 34 | RAG full pipeline (new) |
| `tests/test_rag.py` | varies | RAG unit tests |
| `tests/test_phase5.py` | varies | Model architecture |
| `tests/test_attention.py` | varies | Attention mechanism |
| `tests/test_normalization.py` | varies | RMSNorm |
| `tests/test_rotary.py` | varies | RoPE |
| `tests/test_embeddings.py` | varies | Embedding layer |
| `tests/test_agents.py` | varies | Agent system |
| `tests/test_tools.py` | varies | Agent tools |
| `tests/test_router.py` | varies | Task routing |
| `tests/test_e2e.py` | varies | End-to-end |
| `tests/test_database.py` | varies | SQLite layer |
| `tests/test_security.py` | varies | Security/offline |
| `tests/test_blended_corpus.py` | varies | Corpus blending |
| `tests/test_fineweb_index.py` | varies | FineWeb index |
| `tests/test_fineweb_knowledge.py` | varies | FineWeb knowledge |
| `tests/test_streaming_dataset.py` | varies | Streaming data |
| `tests/test_training_safety.py` | varies | Training safety |
| `tests/test_backend_chat_format.py` | varies | Chat API format |
