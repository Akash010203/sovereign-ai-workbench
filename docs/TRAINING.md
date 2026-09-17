# Model Training Pipeline & Methodology

This document details the training methodology, data engineering, optimization techniques, and execution workflows used to train the custom Sovereign Transformer LLM from scratch.

---

## 1. Overview & Separation of Concerns

The training infrastructure is decoupled from the model architecture:

```
training/
├── dataset.py      # Autoregressive sequence batching & dataset loader
├── trainer.py      # Training loop: AMP, gradient accumulation, logging
├── evaluation.py   # Validation loss, perplexity calculation, BLEU/ROUGE
├── scheduler.py    # Cosine annealing with linear warmup
└── experiments/    # Experiment definitions and configs
```

---

## 2. From-Scratch Random Initialization

Unlike fine-tuning pretrained weights (e.g., Llama-3, Mistral, GPT), this model starts with **random Gaussian parameter initialization**:
- Weights: Initialized with $\mathcal{N}\left(0, \frac{0.02}{\sqrt{2 \cdot N_{\text{layers}}}}\right)$
- Biases: Initialized to zeros
- Normalization (RMSNorm): Scale parameters initialized to $1.0$

```
Random Gaussian Weights
          │
          ▼
   Custom Transformer
          │
          ▼
   Open-Orca Batches
          │
          ▼
 Gradient Backprop (AdamW)
          │
          ▼
 Learned Sovereign LLM Weights
```

---

## 3. Causal Next-Token Objective

The training objective is autoregressive causal next-token prediction using cross-entropy loss:

$$\mathcal{L} = -\frac{1}{T} \sum_{t=1}^{T} \log P(x_t \mid x_1, x_2, \dots, x_{t-1}; \Theta)$$

In code:
```python
# Shift sequences so tokens predict the immediate next token
shift_logits = logits[..., :-1, :].contiguous()
shift_labels = labels[..., 1:].contiguous()

loss = F.cross_entropy(
    shift_logits.view(-1, shift_logits.size(-1)),
    shift_labels.view(-1),
    ignore_index=pad_token_id
)
```

---

## 4. Optimization & Hyperparameters

| Hyperparameter | Value (Exp 1 - Primary 10M) | Value (Exp 2 - Experimental 13M) |
|---|---|---|
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.95$) | AdamW ($\beta_1=0.9, \beta_2=0.95$) |
| **Peak Learning Rate** | $3.0 \times 10^{-4}$ | $2.5 \times 10^{-4}$ |
| **Min Learning Rate** | $3.0 \times 10^{-5}$ | $2.5 \times 10^{-5}$ |
| **Weight Decay** | $0.01$ | $0.01$ |
| **Gradient Clipping** | $1.0$ | $1.0$ |
| **Warmup Steps** | 1,000 steps | 1,200 steps |
| **LR Schedule** | Cosine Annealing | Cosine Annealing |
| **Precision** | PyTorch AMP (`torch.float16` / `bfloat16`) | PyTorch AMP |
| **Batch Size** | 32 (with gradient accumulation) | 16 (with gradient accumulation) |

---

## 5. Execution Workflows

### 1. Pretraining Smoke Test (50 Steps)
```powershell
python scripts/train_model.py --experiment exp1 --max-steps 50 --device cuda
```

### 2. Full Training on Processed OpenOrca
```powershell
python scripts/train_model.py `
    --train-file data/processed/openorca_pretrain.txt `
    --vocab-path tokenizer/vocab/bpe_8k.json `
    --max-steps 25000 `
    --batch-size 32 `
    --device cuda
```

### 3. Instruction Fine-Tuning
```powershell
python scripts/finetune_instruct.py `
    --checkpoint checkpoints/best.pt `
    --max-steps 11250 `
    --batch-size 16 `
    --device cuda
```

### 4. End-to-End Automated Pipeline
```powershell
python scripts/train_pipeline.py --n 90000 --device cuda
```

### 5. RTX 4050 Three-Dataset Blend (80M model)

`scripts/prepare_blended_corpus.py` streams the sources to text files. It
does not cache their complete Arrow/Parquet data in RAM. The default blend is
45% industrial maintenance, 35% NVIDIA SFT reasoning, and 20% OpenOrca
instruction data. VRAM determines the training micro-batch, not the number of
dataset rows; disk capacity and training time determine corpus size.

```powershell
pip install -r requirements.txt

# A practical first corpus: 135k maintenance, 105k NVIDIA, 60k OpenOrca rows.
python scripts/prepare_blended_corpus.py --total-rows 300000

# Train a tokenizer on a representative portion rather than loading the corpus
# into RAM. The control tokens preserve chat boundaries in the blended text.
python scripts/train_tokenizer.py `
    --train-file data/processed/blended_train.txt `
    --output tokenizer/vocab/blended_16k.json `
    --vocab-size 16000 `
    --max-lines 250000 `
    --instruction-format

# 83.1M parameters at 16k vocabulary. The preset uses micro-batch 1 and
# gradient accumulation 16 for a 6 GB RTX 4050.
python scripts/train_model.py `
    --experiment industrial_4050 `
    --model-size industrial-80m `
    --vocab-path tokenizer/vocab/blended_16k.json `
    --train-file data/processed/blended_train.txt `
    --val-file data/processed/blended_val.txt `
    --streaming-data `
    --device cuda
```

To include the approximately 2.1M currently published maintenance rows at the
same blend, request about 4,666,667 total rows. This also selects roughly
1.63M NVIDIA and 0.93M OpenOrca rows, needs substantial disk and takes a long
time on a laptop GPU:

```powershell
python scripts/prepare_blended_corpus.py --total-rows 4666667
```

Review the NVIDIA dataset's per-sample terms before redistributing model
weights: its card notes additional Llama Community License obligations for
some derived models.

#### Offline source download (recommended for unreliable connections)

Download the required source files once; Hugging Face resumes an interrupted
download. Then build the blend with no dataset network reads:

```powershell
python scripts/download_blend_sources.py
python scripts/prepare_blended_corpus.py `
    --total-rows 300000 `
    --source-dir data/raw/hf_sources
```

The default offline download includes the complete NVIDIA `SFT/code` file,
not the entire NVIDIA release. `--all-nemotron` fetches the full ~130 GB
release, but that is not a sensible first run for an 80M model on a 6 GB GPU.
