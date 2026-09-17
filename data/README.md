# Data directory

| folder | purpose |
|---|---|
| `raw/` | Original text exactly as obtained (or authored), untouched. |
| `cleaned/` | Whitespace-normalized, empty lines removed. |
| `processed/` | Length-filtered, split into `train.txt` / `val.txt`, ready for tokenization. |
| `metadata/` | One JSON file per raw source describing its origin, license, and processing notes. |
| `demo/` | Small files used specifically for the SIH live demo (Phase 23). Empty until then. |

Run `python scripts/prepare_data.py` to regenerate `cleaned/`,
`processed/`, and `metadata/` from whatever `.txt` files are in `raw/`.

Nothing is downloaded automatically. Every file placed in `raw/` must
have a corresponding entry recorded in `metadata/` (this is done
automatically by the pipeline script, but the policy is: **no silent
downloads, ever**).

## FineWeb-Edu knowledge ingestion

FineWeb-Edu can be streamed into `data/knowledge/fineweb_edu/` without
materialising the source dataset in RAM.  This is a RAG knowledge-base path;
it creates retrieval chunks and (optionally) embeddings.  It does **not**
train the MiniLLM or change its weights.

```powershell
# Validate the pipeline first; this downloads only enough streamed rows to meet the cap.
python scripts/ingest_fineweb_knowledge.py --target-tokens 100000 --max-source-rows 1000

# Build the requested estimated 1.6B-token knowledge corpus and its embedding shards.
# The sentence-transformer must already be cached locally when --require-neural is used.
python scripts/ingest_fineweb_knowledge.py --target-tokens 1600000000 --embed --require-neural
```

The command checkpoints every 100 source documents and automatically retries
temporary Hugging Face connection/DNS timeouts.  Keep the exact same output
directory and settings when restarting: the checkpoint continues the run.

The pipeline writes JSONL chunk shards, an atomic resume checkpoint, and a
manifest with source/configuration/counts.  “1.6B tokens” is estimated from
normalized *unique source text* characters (not the extra overlap copied into
retrieval chunks) because the streamed source does not provide this project's
custom-tokenizer IDs; retokenize the final text with the training tokenizer if
an exact *training* token budget is required.  The existing
`scripts/prepare_blended_corpus.py` remains the path for mixed training data.
