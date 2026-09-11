# Datasets & Data Lifecycle Management

This document details the data engineering, preprocessing pipelines, training splits, and licensing terms used in the **Sovereign AI Workbench**.

---

## 1. Directory Structure

Data artifacts are strictly partitioned by lifecycle phase:

```
data/
├── raw/         # Raw dataset files or synthetic corpora
├── cleaned/     # Normalized, deduplicated, and filtered text records
├── processed/   # Tokenized sequences, train/val splits (.txt and .jsonl)
├── metadata/    # JSON metadata tracking dataset provenance, row counts, and hashes
└── demo/        # Sample technical documents, engineering manuals, and test files
```

---

## 2. Primary Dataset: Open-Orca

The sovereign LLM utilizes curated subsets from the **Open-Orca** dataset:
- **Hugging Face Repository**: [`Open-Orca/OpenOrca`](https://huggingface.co/datasets/Open-Orca/OpenOrca)
- **Nature of Data**: Multi-turn and single-turn instruction-following dialogues aligned with FLAN and GPT-4 reasoning chains.
- **Sequence Formatting**:
  ```
  <|system|> You are a helpful sovereign engineering assistant.
  <|user|> Explain the operating principle of a centrifugal pump.
  <|assistant|> A centrifugal pump converts rotational kinetic energy from an impeller into hydrodynamic energy...
  ```

---

## 3. Data Processing Pipeline

```
Hugging Face / Local Parquet
             │
             ▼
 `scripts/prepare_openorca.py`
             │
 ┌───────────┴───────────┐
 ▼                       ▼
data/processed/         data/processed/
openorca_pretrain.txt   openorca_instruct.jsonl
(Autoregressive stream) (Supervised prompt-response pairs)
             │
             ▼
 data/metadata/openorca.json
 (Tracks row count, train/val split ratios, provenance)
```

---

## 4. Attribution & Licensing

1. **Open-Orca**: Licensed under MIT / Creative Commons (CC-BY-4.0).
2. **Honesty Ledger**: We explicitly acknowledge Microsoft Research and the Open-Orca community for dataset curation. The dataset provides training examples; the tokenizer, neural network architecture, and training loops are 100% project-authored.
