# Model Architectures & Configurations

The **Sovereign AI Workbench** implements a custom decoder-only Transformer language model from scratch, along with a unified adapter interface for local model integration.

---

## 1. Primary Model: ~10M Parameter Sovereign LLM

The primary model is a compact decoder-only Transformer designed to run on consumer hardware (e.g., laptop RTX 4050 with 6GB VRAM) while demonstrating full architectural integrity.

```
models/custom_minilm/
├── config.py         # Config dataclasses (ModelConfig, TrainConfig)
├── embeddings.py     # Token embedding lookup table
├── rotary.py         # Rotary Position Embeddings (RoPE)
├── normalization.py  # RMSNorm implementation
├── attention.py      # Causal Multi-Head Self-Attention with KV cache
├── ffn.py            # SwiGLU Feed-Forward Network
├── block.py          # Residual Transformer Block
├── model.py          # Complete Decoder-Only Transformer LLM
├── generate.py       # Autoregressive decoding (greedy & top-p/top-k sampling)
└── checkpoint.py     # State dictionary serialization and checkpoint loading
```

### Architectural Parameters
| Hyperparameter | Primary Configuration | Experimental Variant |
|---|---|---|
| **Parameter Count** | **~10.2 Million** | **~13.5 Million** |
| **Layers ($N_{\text{layers}}$)** | 8 | 8 |
| **Hidden Dimension ($d_{\text{model}}$)** | 256 | 384 |
| **Attention Heads ($N_{\text{heads}}$)** | 8 | 6 |
| **Head Dimension ($d_{\text{head}}$)** | 32 | 64 |
| **FFN Dimension ($d_{\text{ff}}$)** | 1024 | 1536 |
| **Vocabulary Size ($V$)** | 8,192 | 8,192 |
| **Max Context Length** | 512 | 512 |
| **Positional Encoding** | Rotary Position Embeddings (RoPE) | Rotary Position Embeddings (RoPE) |
| **Activation Function** | SwiGLU | SwiGLU |
| **Normalization** | RMSNorm ($\epsilon=10^{-6}$) | RMSNorm ($\epsilon=10^{-6}$) |
| **Weight Tying** | Yes (LM Head shares $W_e$) | Yes |

---

## 2. Component Mathematical Formulations

### A. RMSNorm
$$\text{RMSNorm}(x) = \frac{x}{\sqrt{\frac{1}{d}\sum_{i=1}^d x_i^2 + \epsilon}} \odot \gamma$$

### B. Rotary Position Embeddings (RoPE)
Given vector $x \in \mathbb{R}^{d}$, RoPE applies orthogonal rotation matrices $R_{\Theta, m}$ depending on token index $m$:
$$\mathbf{q}_m = R_{\Theta, m} W_q \mathbf{x}_m, \quad \mathbf{k}_n = R_{\Theta, n} W_k \mathbf{x}_n$$
This allows relative position encoding via inner products:
$$\langle \mathbf{q}_m, \mathbf{k}_n \rangle = f(\mathbf{x}_m, \mathbf{x}_n, m - n)$$

### C. SwiGLU Feed-Forward Network
$$\text{SwiGLU}(x) = \left( \text{Swish}(x W_{\text{gate}}) \otimes (x W_{\text{up}}) \right) W_{\text{down}}$$
where $\text{Swish}(z) = z \cdot \sigma(z)$.

---

## 3. Pluggable Adapters & Model Registry

In addition to the from-scratch custom LLM, the workbench supports pluggable local models via `models/adapters/`:
- `CustomLLMAdapter`: Runs native PyTorch inference using trained weights.
- `OllamaAdapter`: Connects to local Ollama runtime for open-weight models (e.g., Mistral, Phi-3).
- `VisionAdapter`: Connects to local multimodal models (e.g., LLaVA) for image and visual diagram understanding.

The `models/registry.py` manages lifecycle, health checks, and fallback mechanisms across all models.
