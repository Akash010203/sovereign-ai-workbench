# tests/ (root-level)

Module-specific tests that need `torch` live here (mirroring
`tokenizer/tests/`, which doesn't need torch and stays separate).

## Phase 4 — neural network foundations (current contents)

- `test_embeddings.py` — token embedding lookup (`models/custom_minilm/embeddings.py`)
- `test_normalization.py` — RMSNorm (`models/custom_minilm/normalization.py`)
- `test_rotary.py` — RoPE (`models/custom_minilm/rotary.py`)
- `test_attention.py` — causal multi-head self-attention (`models/custom_minilm/attention.py`)

Every file uses `torch = pytest.importorskip("torch")` at the top, so
if torch isn't installed the file is cleanly **skipped**, not
reported as a failure. Run:

```
python -m pytest tests/ -v
```

Also reserved for later phases: Phase 6 training-loop smoke tests,
Phase 21 end-to-end integration tests.
