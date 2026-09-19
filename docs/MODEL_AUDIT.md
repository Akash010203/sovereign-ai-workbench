# SovereignAI — Model Audit

> Generated: 2026-09-19 by forensic audit.

## Custom MiniLLM — Classification: Category B

**Genuine custom architecture with self-trained weights.**

### Architecture Verification

| Component | Implementation | File | Status |
|---|---|---|---|
| Token Embedding | `nn.Embedding(vocab_size, d_model)` | `models/custom_minilm/embeddings.py` | ✅ EXISTS |
| Causal Self-Attention | Multi-head with RoPE, causal mask | `models/custom_minilm/attention.py` | ✅ EXISTS |
| RoPE Positional Encoding | Rotary position embedding | `models/custom_minilm/rotary.py` | ✅ EXISTS |
| RMSNorm | Root Mean Square Layer Normalization | `models/custom_minilm/normalization.py` | ✅ EXISTS |
| SwiGLU FFN | Gated feed-forward with SiLU activation | `models/custom_minilm/ffn.py` | ✅ EXISTS |
| Transformer Block | Pre-norm attention + FFN + residuals | `models/custom_minilm/block.py` | ✅ EXISTS |
| Full Model | Block stack + final norm + LM head | `models/custom_minilm/model.py` | ✅ EXISTS |
| Weight Tying | LM head shares embedding weights | `model.py:145` | ✅ EXISTS |
| Weight Init | GPT-2 style Normal(0, 0.02) + residual scaling | `model.py:150-174` | ✅ EXISTS |
| Loss | Cross-entropy with ignore_index=-100 | `model.py:221-256` | ✅ EXISTS |
| Checkpoint | Save/load with config + optimizer state | `models/custom_minilm/checkpoint.py` | ✅ EXISTS |
| Generation | Top-k/top-p sampling with temperature | `models/custom_minilm/generate.py` | ✅ EXISTS |

### Model Size Presets

| Preset | d_model | n_layers | n_heads | d_ff | max_seq_len | ~Params |
|---|---|---|---|---|---|---|
| Small | 256 | 6 | 4 | 683 | 128 | ~5M |
| Medium | 384 | 8 | 6 | 1024 | 256 | ~12M |
| Industrial 80M | 768 | 12 | 12 | 1536 | 256 | ~83M |

### Hidden API/Model Search

| Search Term | Found? | Location | Classification |
|---|---|---|---|
| `openai` | No | — | — |
| `claude` | No | — | — |
| `gemini` | No | — | — |
| `api.openai.com` | No | — | — |
| `anthropic` | No | — | — |
| `requests.post` | No | — | — |
| `httpx` | No | — | — |

**No hidden external API calls detected in the model or inference code.**

### Adapter Classification

| Adapter | File | Classification | Status |
|---|---|---|---|
| CustomMiniLLMAdapter | `models/adapters/minilm_adapter.py` | B — Custom arch + self-trained weights | ✅ Active |
| OpenWeightAdapter | `models/adapters/openweight_adapter.py` | D — Pretrained wrapper (Ollama) | ❌ DELETED in working tree |

### Training Evidence

| Experiment | Steps | Final Train Loss | Final Val Loss | Checkpoint |
|---|---|---|---|---|
| exp1_smoke | ~50 | — | — | `exp1_smoke_step00050_final.pt` |
| exp4_openorca | 30,000 | — | — | `exp4_openorca_step30000_final.pt` |
| industrial_80m_4050 | 20,000 | 2.37 | 3.57 (perplexity 35.5) | `industrial_80m_4050_step20000_final.pt` |

### RAG Embedding Model — Classification: Category C

| Component | Model | Classification | Purpose |
|---|---|---|---|
| SentenceTransformerEmbedder | all-MiniLM-L6-v2 | C — Local pretrained model | RAG vector embedding ONLY |
| TFIDFEmbedder | Custom hash TF-IDF | F — Rule-based | Zero-dependency fallback |

**Honesty note**: The SentenceTransformer embedding model is NOT the project's custom LLM. It is a separate pretrained model used exclusively for the RAG vector search layer. This is documented in the project's honesty ledger.
