# training/ (reserved)

Empty until **Phase 6** (see `docs/ARCHITECTURE.md` §11).

Planned contents: `dataset.py` (torch `Dataset` over the tokenized
demo corpus), `trainer.py` (the real autoregressive training loop —
forward pass, cross-entropy loss, backward pass, AdamW, gradient
clipping, LR schedule, checkpointing), `scheduler.py`,
`evaluation.py`, `experiments/` (the staged 10-50 / 100-500 / longer
runs described in the build contract).
