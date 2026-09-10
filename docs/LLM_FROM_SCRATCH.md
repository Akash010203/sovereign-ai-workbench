# LLM From Scratch — How the SovereignAI MiniLLM Works

> **Audience**: Akash (B.Tech CS/AI-ML, SIH 2026, Problem Statement SIH 117)  
> **Purpose**: Judge-explainable walkthrough of every component built from scratch

---

## Table of Contents

1. [What is a Language Model?](#1-what-is-a-language-model)
2. [What Are Tokens?](#2-what-are-tokens)
3. [The Full Model Architecture](#3-the-full-model-architecture)
4. [Token Embeddings](#4-token-embeddings)
5. [Rotary Position Encoding (RoPE)](#5-rotary-position-encoding-rope)
6. [Causal Multi-Head Self-Attention](#6-causal-multi-head-self-attention)
7. [RMSNorm](#7-rmsnorm)
8. [SwiGLU Feed-Forward Network](#8-swiglu-feed-forward-network)
9. [The Transformer Block](#9-the-transformer-block)
10. [The Language Model Head & Weight Tying](#10-the-language-model-head--weight-tying)
11. [Cross-Entropy Loss](#11-cross-entropy-loss)
12. [Parameter Count Breakdown](#12-parameter-count-breakdown)
13. [What "From Scratch" Means for This Project](#13-what-from-scratch-means-for-this-project)
14. [What This Model Can and Cannot Do](#14-what-this-model-can-and-cannot-do)

---

## 1. What is a Language Model?

A language model is a function that takes a sequence of tokens and
predicts which token is most likely to come next.

```
Input:  "The inspection report was forwarded"
Output: probability over every token in the vocabulary for what comes next
        e.g., " to" → 0.42,  " by" → 0.11,  " from" → 0.08, ...
```

During **training**, the model is shown billions of (input → next_token)
pairs from real text and nudged — via gradient descent — to improve its
predictions. During **generation**, the model is run autoregressively:
it predicts the next token, appends it, predicts again, and so on.

This project's MiniLLM is a decoder-only autoregressive language model
(same family as GPT-2, LLaMA, Mistral — just much smaller).

---

## 2. What Are Tokens?

A token is the basic unit the model operates on. Raw text (a string of
Unicode characters) is not directly usable by a neural network — numbers
are needed.

The **tokenizer** (Phase 3) converts text to integers and back:

```
"boiler pressure"  →  encode()  →  [42, 17, 388, 251]
[42, 17, 388, 251] →  decode()  →  "boiler pressure"
```

This project uses a **byte-level BPE tokenizer** built from scratch
(no tokenizer library). See `docs/TOKENIZER.md` for the full explanation.

Vocabulary size = **600** tokens (for the demo corpus).  
At inference time, the model outputs a probability over all 600 tokens
at each step and picks the highest (or samples from the distribution).

---

## 3. The Full Model Architecture

```
Input text
   │
   ▼
 tokenizer.encode()
   │  [B, T]  — batch of integer sequences
   │
   ▼
 TokenEmbedding          [B, T] → [B, T, d_model=256]
   │
   ▼
 TransformerBlock × 6    [B, T, C] → [B, T, C]  (each block)
   │
   ├─ RMSNorm
   ├─ CausalSelfAttention + RoPE
   ├─ residual (+)
   ├─ RMSNorm
   ├─ SwiGLUFFN
   └─ residual (+)
   │
   ▼
 Final RMSNorm           [B, T, C] → [B, T, C]
   │
   ▼
 LM Head (tied weight)   [B, T, C] → [B, T, vocab_size=600]
   │
   ▼
 Logits  →  softmax → probability distribution
              or
         →  cross_entropy(targets) → loss → backward()
```

**Shape legend** (used throughout all files in this project):

| Symbol | Meaning |
|--------|---------|
| `B` | Batch size (number of sequences processed in parallel) |
| `T` | Sequence length (number of tokens in one sequence) |
| `C` | `d_model` = embedding / hidden dimension = **256** |
| `H` | `n_heads` = number of attention heads = **4** |
| `D` | `head_dim` = C / H = 256 / 4 = **64** |
| `vocab_size` | Total vocabulary tokens = **600** |

---

## 4. Token Embeddings

**File**: `models/custom_minilm/embeddings.py`

Each token ID is an integer in `[0, vocab_size)`.  The embedding layer
is a lookup table of shape `[vocab_size, d_model]`.

```
input_ids: [B, T] (integers)
    │
    ▼  lookup row i from the [vocab_size × C] weight matrix
    │
output: [B, T, C]  (continuous floating-point vectors)
```

Every row of this table is a **learned vector representation** of one
token. At the start of training the table is random noise. After
training, similar tokens (like "pressure" and "stress") end up with
similar vectors because they appear in similar contexts.

**Why use `nn.Embedding`?**  
A lookup table is exactly what `nn.Embedding` is — there is no
mathematical insight to reinvent here. What is from scratch is
everything that *uses* these vectors: the attention math, the
normalization, the positional encoding, and the FFN.

---

## 5. Rotary Position Encoding (RoPE)

**File**: `models/custom_minilm/rotary.py`

### Why position matters

Attention has no built-in sense of order. Without position information,
the model sees every pair of tokens as equally near each other —
"pump failed after valve" and "valve failed after pump" would be
identical to it.

### How RoPE works

Instead of adding a positional vector to embeddings (the original 2017
approach), RoPE **rotates** the query and key vectors by an angle that
is proportional to the token's position in the sequence.

For each adjacent pair of dimensions in a query/key vector, RoPE treats
the pair as a 2-D point and rotates it:

```
x_new[..., 2i  ] = x[..., 2i  ] × cos(θ_i × pos) - x[..., 2i+1] × sin(θ_i × pos)
x_new[..., 2i+1] = x[..., 2i  ] × sin(θ_i × pos) + x[..., 2i+1] × cos(θ_i × pos)
```

where `θ_i = 1 / (10000^(2i / head_dim))`.

Key properties:
- **Rotation preserves vector length** — no distortion of magnitudes.
- **Relative positions are captured** — the dot product `q · k` depends
  on `position_q - position_k`, exactly what attention needs.
- **No extra parameters** — the rotation angles are precomputed
  constants, not learned.

---

## 6. Causal Multi-Head Self-Attention

**File**: `models/custom_minilm/attention.py`

### The three questions every token asks

For each token at position `t`, attention computes:
- **Query** (`q`): "What am I looking for?"  
- **Key** (`k`): "What information do I offer to others?"  
- **Value** (`v`): "What content do I carry?"

### The math (step by step)

```
1. Project input x [B, T, C] to q, k, v  (each [B, T, C])
2. Apply RoPE to q and k
3. Reshape to [B, H, T, D]  (H heads, each of D dimensions)
4. Attention scores = (q @ k^T) / sqrt(D)      → [B, H, T, T]
5. Apply causal mask: future positions → -inf   → [B, H, T, T]
6. Softmax over key dimension → attention weights → [B, H, T, T]
7. Weighted sum of values: weights @ v           → [B, H, T, D]
8. Concatenate heads: reshape to [B, T, C]
9. Output projection: W_o × [B, T, C]           → [B, T, C]
```

### Why multiple heads?

Different heads can specialize in different patterns. One head might
learn to attend to the most recent noun, another to the sentence's verb.
The `head_dim = D = 64` gives each head enough expressiveness while
keeping total computation the same as a single 256-dimensional
attention.

### The causal mask

```
Position:   0  1  2  3
         0 [1, 0, 0, 0]    token 0 sees only itself
         1 [1, 1, 0, 0]    token 1 sees tokens 0 and 1
         2 [1, 1, 1, 0]    token 2 sees tokens 0, 1, 2
         3 [1, 1, 1, 1]    token 3 sees all past tokens
```

Positions marked `0` are set to `-inf` before softmax, so they receive
zero attention weight. This prevents the model from "cheating" during
training by looking at the answer.

---

## 7. RMSNorm

**File**: `models/custom_minilm/normalization.py`

### Why normalize?

As a Transformer deepens, the scale of activations can drift and make
gradients explode or vanish. Normalization re-scales activations before
each sub-layer.

### The formula

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) × weight
```

- `mean(x²)` is the **root mean square** of `x`'s values along the last dimension.
- Dividing by it re-scales `x` to have RMS ≈ 1.
- `weight` is a learned per-channel scale (initialized to 1).
- `ε = 1e-5` prevents division by zero.

### Why RMSNorm instead of LayerNorm?

Standard LayerNorm also subtracts the mean (re-centers to zero).
RMSNorm skips the re-centering step — it's cheaper and works equally
well for Transformers. It's used by LLaMA, Mistral, and this project.

---

## 8. SwiGLU Feed-Forward Network

**File**: `models/custom_minilm/ffn.py`

### What the FFN does

After attention mixes information *across* positions, the FFN processes
each position *independently* with the same learned weights.
Despite no cross-position communication, the FFN holds the majority
of the model's parameters and capacity.

### Classic vs SwiGLU

```
Classic FFN:  output = ReLU(x W₁) W₂
SwiGLU FFN:   output = (SiLU(x W_gate) ⊙ x W_up) W_down
```

Where:
- `SiLU(z) = z × sigmoid(z)` — the Swish activation
- `⊙` — element-wise multiplication (the "gate")
- The gate learns *which neurons to suppress*
- The up-projection learns *what to amplify*

SwiGLU trains faster to the same loss than ReLU at matched parameter
count. It's used by LLaMA, PaLM, and this project.

### Why d_ff = 683?

SwiGLU has **3** weight matrices (gate, up, down) instead of 2.
To match the FLOPs of classic 4× hidden-size FFN:

```
classic : d_ff = 4 × 256 = 1024   (2 matrices × 1024)
SwiGLU  : d_ff = (4 × 2/3) × 256 ≈ 683  (3 matrices × 683 ≈ 2 × 1024)
```

---

## 9. The Transformer Block

**File**: `models/custom_minilm/block.py`

One block = two sub-layers, each wrapped in pre-norm + residual:

```
x  ──┬── RMSNorm ──> CausalSelfAttention ──> (+) ──> y
     │                                        ↑
     └────────────────────────────────────────┘

y  ──┬── RMSNorm ──> SwiGLUFFN ────────────> (+) ──> z
     │                                        ↑
     └────────────────────────────────────────┘
```

**Pre-norm**: the sub-layer receives a normalized input. This makes
early training more stable because the gradients from random
initialization don't cause immediate instability.

**Residual connection**: the sub-layer's *change* (delta) is added to
the input, not replacing it. This provides a shortcut path for gradients
to flow directly from the loss all the way back to the earliest layers —
preventing the vanishing gradient problem in deep networks.

The model stacks **6 blocks** (controlled by `config.n_layers`).

---

## 10. The Language Model Head & Weight Tying

**File**: `models/custom_minilm/model.py`

The LM head is a single `nn.Linear(d_model, vocab_size, bias=False)` that
projects the hidden state at each position to a score for every token in
the vocabulary.

### Weight tying

```python
self.lm_head.weight = self.token_embedding.weight
```

The LM head's weight matrix is set to be the **same tensor** as the
token embedding matrix. Both refer to the same block of memory.

**Why?**  
- Saves ~`vocab_size × d_model` = 600 × 256 = 153,600 parameters.
- Tokens that the model predicts frequently become "closer" in the
  embedding space, which aids training.
- Used by GPT-2, LLaMA, and most small language models.

---

## 11. Cross-Entropy Loss

For autoregressive language modelling, every token predicts the next one.

```
Input  sequence:  [<bos>, t1,  t2,  t3,  t4]
Target sequence:  [t1,    t2,  t3,  t4,  <eos>]

At position 0: model sees <bos>, must predict t1
At position 1: model sees t1, must predict t2
...
```

Cross-entropy loss for one position:

```
loss = -log(p_model(correct_token))
```

The total loss is the mean over all non-masked positions. A well-
calibrated model at random initialization (uniformly guessing among
600 tokens) should produce `loss ≈ ln(600) ≈ 6.4`. After training,
the loss decreases, indicating the model has learned to make better
predictions.

---

## 12. Parameter Count Breakdown

With the default config (`d_model=256, n_layers=6, n_heads=4,
d_ff=683, vocab_size=600`):

| Component | Formula | Count |
|-----------|---------|-------|
| Token embedding | `vocab_size × d_model` | 153,600 |
| Per block: Q/K/V/O projections | `4 × d_model²` | 262,144 |
| Per block: SwiGLU (gate+up+down) | `3 × d_model × d_ff` | 524,544 |
| Per block: 2× RMSNorm | `2 × d_model` | 512 |
| **Per block total** | | **787,200** |
| **6 blocks total** | `6 × 787,200` | **4,723,200** |
| Final RMSNorm | `d_model` | 256 |
| LM head | tied — 0 extra | 0 |
| **TOTAL** | | **≈ 4,877,056** |

This fits comfortably into 6 GB VRAM for training and inference.

---

## 13. What "From Scratch" Means for This Project

### Genuinely from scratch (no library does the math):
- Byte-level BPE tokenizer (`tokenizer/tokenizer.py`, `trainer.py`)
- RMSNorm formula (`normalization.py`)
- RoPE rotation math (`rotary.py`)
- Attention score / softmax / weighted-sum (`attention.py`)
- Causal mask construction (`attention.py`)
- SwiGLU gating (`ffn.py`)
- Model assembly, weight tying, parameter counting (`model.py`)
- Cross-entropy loss call with target shifting (`model.py`)

### Uses PyTorch primitives (explicitly allowed by the build contract):
- `nn.Embedding` — a lookup table primitive, not a Transformer
- `nn.Linear` — matrix multiply primitive
- `nn.Dropout` — a regularization primitive
- `torch.softmax` — numerical function, not a Transformer
- `F.silu` — activation function, not a Transformer
- `torch.nn.functional.cross_entropy` — loss function primitive

### What is explicitly NOT used:
- `nn.Transformer`
- `nn.MultiheadAttention`
- `F.scaled_dot_product_attention`
- Any pretrained weights (Hugging Face, OpenAI, Meta, etc.)
- Any cloud AI API

---

## 14. What This Model Can and Cannot Do

**CAN:**
- Demonstrate that every component of a Transformer was built from scratch
- Train on the project's custom corpus and show loss decreasing
- Generate text autoregressively after training
- Prove air-gapped, fully local operation
- Serve as the "custom model" slot in the multi-model SovereignAI platform

**CANNOT:**
- Compete with GPT-4, Claude, LLaMA-3, or any frontier LLM
- Produce coherent long-form text on arbitrary topics
- Perform complex reasoning

**Why it's honest to call it an educational MiniLLM:**  
A 5M-parameter model trained on ~66 lines of text cannot be intelligent
in any meaningful sense. Its purpose is to prove the architecture and
training pipeline are genuine — which is exactly what SIH 117 requires.
The SovereignAI platform's production capability comes from routing
tasks to appropriately-sized locally-hosted open-weight models (Phase 8+).

---

*Document written by Akash — SovereignAI, SIH 2026, Problem Statement SIH 117.*
