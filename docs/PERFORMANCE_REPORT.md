# SovereignAI — Performance Report

> Generated: 2026-09-19 by forensic audit.

## Hardware Context

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 4050 Laptop GPU |
| VRAM | 6 GB |
| System RAM | 16 GB |
| OS | Windows 11 |

## Model Performance

### Custom MiniLLM — Industrial 80M (~83M params)

| Metric | Value | Notes |
|---|---|---|
| Training steps completed | 20,000 | On blended corpus |
| Final training loss | 2.375 | Cross-entropy |
| Final validation loss | 3.570 | Held-out set |
| Final validation perplexity | 35.5 | exp(val_loss) |
| Step duration | ~0.9 seconds | With AMP + gradient accumulation |
| Gradient norm (final) | 1.75 | Stable, no exploding gradients |
| Total training time | ~5 hours (est.) | 20K steps × 0.9s |

### Training Convergence

```
Step     1: loss = 9.816, grad_norm = 10.94, lr = 4e-7
Step 20000: loss = 2.375, grad_norm = 1.75,  lr = 2e-5
```

**Assessment**: The model shows clear learning signal with a ~4x loss reduction. Validation perplexity of 35.5 is reasonable for an 83M parameter model on a 1.6B token corpus — this is well below random (which would be ~log(vocab_size) ≈ 9.7). The model has learned meaningful language patterns.

### Model Size vs VRAM Budget

| Preset | Parameters | VRAM Usage (est.) | Fits RTX 4050? |
|---|---|---|---|
| Small (5M) | 4.88M | ~100 MB | ✅ YES |
| Medium (12M) | ~12M | ~250 MB | ✅ YES |
| Industrial 80M | ~83M | ~2 GB (inference) / ~4 GB (training w/ AMP) | ✅ YES |

## RAG Performance

### Embedding Speed

| Embedder | Latency (single) | Latency (batch of 100) | Quality |
|---|---|---|---|
| SentenceTransformer | ~200ms (first), ~5ms (cached) | ~50ms | High (neural) |
| TF-IDF | <1ms | ~10ms | Lower (statistical) |

### Retrieval Accuracy (from regression tests)

| Query | Correct Source Retrieved? | Score |
|---|---|---|
| "maximum operating temperature" | ✅ unit_a_specifications.txt | High |
| "inspection interval for Pump B" | ✅ pump_b_maintenance.txt | High |
| "emergency shutdown procedure" | ✅ emergency_shutdown.txt | High |
| "bolt torque for flange" | ✅ engineering_notes.txt | High |
| "quantum chromodynamics" (negative) | ✅ No results at min_score=0.8 | N/A |

### Index Sizes

| Index | Chunks | Search Time (est.) | Memory |
|---|---|---|---|
| User knowledge (10,975 chunks) | 10,975 | ~10ms (brute force) | ~90 MB in RAM |
| FineWeb 1.6B (5.2M chunks) | 5,240,427 | ~seconds per shard | ~77 MB per shard (mmap) |

## Bottlenecks & Recommendations

1. **RAG search is brute-force**: For 10K chunks this is fine (<50ms). For 5.2M FineWeb chunks, consider adding FAISS or HNSW indexing.
2. **Model context is 256 tokens**: This limits how much RAG context can be used. Consider extending to 512 tokens if VRAM allows.
3. **SentenceTransformer loads 3× per validation**: Singleton pattern would reduce startup time.
4. **JSON index persistence**: For indexes >100K chunks, consider SQLite or binary format instead of JSON.
