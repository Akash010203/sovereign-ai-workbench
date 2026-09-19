# SOVEREIGNAI SYSTEM HEALTH

> Generated: 2026-09-19 00:35:40

## Component Status

| Component | Status | Duration | Details |
|---|---|---|---|
| environment | ✅ PASS | 1.7s |  |
| repository | ✅ PASS | 0.1s |  |
| dataset | ✅ PASS | 0.0s |  |
| tokenizer | ✅ PASS | 0.1s |  |
| model | ✅ PASS | 1.1s |  |
| checkpoints | ✅ PASS | 0.0s |  |
| rag | ✅ PASS | 7.6s |  |
| fineweb | ✅ PASS | 0.0s |  |
| database | ✅ PASS | 0.0s |  |
| router | ✅ PASS | 0.0s |  |
| security | ✅ PASS | 0.0s |  |
| docx | ✅ PASS | 0.1s | python-docx available |
| xlsx | ✅ PASS | 0.2s |  |
| pptx | ✅ PASS | 0.1s |  |

## Summary

- **Passed**: 14/14
- **Failed**: 0/14

## Evidence

### environment (PASS)
```json
{
  "python_version": "3.14.3 (tags/v3.14.3:323c59a, Feb  3 2026, 16:04:56) [MSC v.1944 64 bit (AMD64)]",
  "platform": "win32",
  "cwd": "C:\\Users\\akash\\Videos\\Captures\\sovereign-ai-workbench",
  "torch_version": "2.11.0+cu128",
  "cuda_available": true,
  "gpu_name": "NVIDIA GeForce RTX 4050 Laptop GPU",
  "vram_gb": 6.0
}
```

### repository (PASS)
```json
{
  "commit": "3d5fdd80081adc610e207880a24d22911a5407d6",
  "dirty_file_count": 64
}
```

### dataset (PASS)
```json
{
  "train.txt": {
    "exists": true,
    "size": 3906
  },
  "val.txt": {
    "exists": true,
    "size": 850
  },
  "blended_train.txt": {
    "exists": true,
    "size_gb": 3.06
  }
}
```

### tokenizer (PASS)
```json
{
  "demo_bpe_vocab.json": {
    "vocab_size": 600,
    "test_encode_len": 17,
    "roundtrip_ok": true
  },
  "bpe_8k.json": {
    "vocab_size": 8000,
    "test_encode_len": 12,
    "roundtrip_ok": true
  },
  "blended_16k.json": {
    "vocab_size": 16000,
    "test_encode_len": 8,
    "roundtrip_ok": true
  }
}
```

### model (PASS)
```json
{
  "small": {
    "params": 8819456,
    "params_m": 8.8,
    "output_shape": [
      1,
      16,
      16000
    ]
  },
  "industrial_80m": {
    "params": 83086080,
    "params_m": 83.1,
    "output_shape": [
      1,
      16,
      16000
    ]
  }
}
```

### checkpoints (PASS)
```json
{
  "count": 452,
  "total_gb": 117.1,
  "industrial_80m_best_exists": true,
  "industrial_80m_best_size_mb": 951.0
}
```

### rag (PASS)
```json
{
  "embedder_type": "SentenceTransformerEmbedder",
  "embedding_dim": 384,
  "chunks_ingested": 334,
  "index_size": 334,
  "retrieval_results": 3,
  "top_result_source": "unit_a_specifications.txt",
  "top_result_score": 0.7806,
  "top_result_text_preview": "The maximum operating temperature of Unit A is 180\u00b0C. This limit was established\nduring the original",
  "negative_test_results": 0
}
```

### fineweb (PASS)
```json
{
  "chunks": 5240427,
  "embedding_dim": 384,
  "source_docs": 1319405,
  "chunk_shards": 105,
  "embedding_shards": 105
}
```

### database (PASS)
```json
{
  "exists": true,
  "size_kb": 172.0,
  "connection": "OK"
}
```

### router (PASS)
```json
{
  "What is the inspection interval?": {
    "category": "engineering_document",
    "model_name": "none",
    "reason": "rule-based routing: category='engineering_document' \u2192 no model available from ['"
  },
  "Calculate 2 + 2": {
    "category": "calculation",
    "model_name": "none",
    "reason": "rule-based routing: category='calculation' \u2192 no model available from ['custom_mi"
  },
  "Write a Python function": {
    "category": "coding",
    "model_name": "none",
    "reason": "rule-based routing: category='coding' \u2192 no model available from ['custom_minilm_"
  }
}
```

### security (PASS)
```json
{
  "offline_check": {
    "timestamp": "2026-09-18T19:05:40.321244+00:00",
    "offline": true,
    "internet_reachable": false,
    "tested_hosts": [],
    "verdict": "NO OUTBOUND PROBE"
  }
}
```
