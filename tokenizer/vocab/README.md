# tokenizer/vocab/

Trained tokenizer files land here as JSON (produced by
`scripts/train_tokenizer.py` or `ByteLevelBPETokenizer.save()`).
`demo_bpe_vocab.json` is the tokenizer trained on the tiny synthetic
demo corpus in Phase 3 — regenerate it any time with:

```
python scripts/prepare_data.py
python scripts/train_tokenizer.py
```
