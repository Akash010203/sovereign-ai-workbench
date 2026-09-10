# models/

## `custom_minilm/` — the from-scratch Transformer

**Phase 4 (done):** the independently-testable building blocks, sized
by the shared `MiniLLMConfig` (~6-7M parameters once assembled):

| File | What it implements |
|---|---|
| `config.py` | `MiniLLMConfig` — every size/hyperparameter, one place |
| `embeddings.py` | `TokenEmbedding` — token ID -> dense vector lookup |
| `normalization.py` | `RMSNorm`, written from raw tensor ops (no `nn.LayerNorm`) |
| `rotary.py` | `RotaryEmbedding` (RoPE), written from raw tensor ops |
| `attention.py` | `CausalSelfAttention`, written from raw matmul/softmax/mask (no `nn.MultiheadAttention`, no `F.scaled_dot_product_attention`) |

Unit tests: `tests/test_embeddings.py`, `tests/test_normalization.py`,
`tests/test_rotary.py`, `tests/test_attention.py`.

**Still reserved** (Phases 5-7): `ffn.py` (SwiGLU), `block.py`
(residual + norm + attention + FFN), `model.py` (full assembled
MiniLLM + LM head + weight tying), `train.py`, `generate.py`,
`checkpoint.py`.

## `adapters/` (reserved — Phase 8)
`ModelProvider` interface + adapters wrapping individual models.

## `vision/` (reserved — Phase 14)
Local vision model wrapper.

## `registry.py` (reserved — Phase 8-9)
Runtime model registry used by the router.

No placeholder code is added ahead of its phase — see the "no fake
implementations" rule in the project's build contract.
