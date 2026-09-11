# Custom Tokenizer Architecture & Implementation

The **Sovereign AI Workbench** features a completely custom, from-scratch **Byte-Pair Encoding (BPE)** tokenizer implemented in Python with zero dependency on external tokenization libraries (such as Hugging Face `tokenizers` or OpenAI `tiktoken`).

---

## 1. Why Tokenization from Scratch?

A language model does not process arbitrary Unicode strings directly. It maps discrete numerical token IDs to vector embeddings. In a sovereign, air-gapped environment:
1. **Full Transparency**: Every merge rule, special token, and vocabulary index is fully inspectable and deterministic.
2. **Offline Independence**: The tokenizer does not require downloading external pre-compiled binaries or pre-trained vocabularies.
3. **Exact Embedding Alignment**: The embedding weight matrix in the Transformer ($W_e \in \mathbb{R}^{V \times d_{\text{model}}}$) directly reflects the vocabulary size $V$ produced by this custom tokenizer.

---

## 2. Tokenizer Architecture

The custom tokenizer is organized under `tokenizer/`:

```
tokenizer/
├── tokenizer.py    # Core BPE encode() and decode() logic
├── trainer.py      # Vocabulary trainer: frequency counting, pair merges
├── vocab/          # Saved vocabulary files (.json)
│   ├── bpe_8k.json # Production 8K vocabulary
│   └── demo_bpe_vocab.json
└── tests/          # Unit test suite for tokenization accuracy
```

---

## 3. Byte-Level BPE Algorithm

### Step 1: Byte-Level Fallback
The tokenizer initializes with all 256 individual byte values ($0x00$ to $0xFF$). This guarantees **zero Out-Of-Vocabulary (OOV) tokens**—any arbitrary character, emoji, or binary sequence can be encoded.

### Step 2: Special Tokens
Special reserved tokens are injected at the front of the vocabulary:
- `<|pad|>`: Padding token (ID: `0`)
- `<|bos|>`: Beginning of sequence (ID: `1`)
- `<|eos|>`: End of sequence (ID: `2`)
- `<|unk|>`: Unknown token fallback (ID: `3`)
- `<|system|>`: System prompt delimiter
- `<|user|>`: User input delimiter
- `<|assistant|>`: Assistant response delimiter

### Step 3: Training via Iterative Pair Merging
1. Initialize the vocabulary with special tokens and all 256 byte tokens.
2. Scan the training corpus (e.g., Open-Orca or domain text) and compute the frequency of all adjacent token pairs $(t_i, t_{i+1})$.
3. Find the most frequent adjacent pair:
   $$\text{pair}^* = \arg\max_{(t_a, t_b)} \text{Count}(t_a, t_b)$$
4. Merge $\text{pair}^*$ into a new combined token, assign it a new token ID, and record the merge rule.
5. Repeat steps 2–4 until the desired vocabulary size $V$ (e.g., $1,000$, $4,096$, or $8,192$) is reached.

---

## 4. Usage & CLI

### Training the Tokenizer
```powershell
python scripts/train_tokenizer.py --train-file data/processed/openorca_pretrain.txt --output tokenizer/vocab/bpe_8k.json --vocab-size 8192
```

### Python API
```python
from tokenizer.tokenizer import BPETokenizer

# Load trained vocabulary
tokenizer = BPETokenizer.load("tokenizer/vocab/bpe_8k.json")

# Encode text
token_ids = tokenizer.encode("Sovereign AI Workbench ensures total data privacy.")
print("Token IDs:", token_ids)

# Decode back to text
decoded_text = tokenizer.decode(token_ids)
print("Decoded:", decoded_text)
```

---

## 5. Validation & Tests

The tokenizer includes comprehensive automated unit tests in `tokenizer/tests/`:
```powershell
python -m pytest tokenizer/tests -v
```
Tests verify:
- Round-trip fidelity: `decode(encode(text)) == text`
- Special token preservation and handling
- Handling of arbitrary Unicode, edge cases, whitespace, and numerical digits.
