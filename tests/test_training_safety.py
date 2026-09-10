"""Regression tests for checkpoint and SFT safeguards."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def test_instruction_dataset_keeps_response_labels_for_long_prompt(tmp_path):
    from scripts.finetune_instruct import InstructDataset
    from tokenizer.tokenizer import ByteLevelBPETokenizer

    path = tmp_path / "long.jsonl"
    path.write_text(json.dumps({
        "system": "system " * 100,
        "question": "question " * 100,
        "response": "the answer is retained",
    }) + "\n", encoding="utf-8")
    tokenizer = ByteLevelBPETokenizer.load(ROOT / "tokenizer/vocab/demo_bpe_vocab.json")

    inputs, targets = InstructDataset(path, tokenizer, max_seq_len=16)[0]

    assert inputs.shape == targets.shape == (16,)
    assert (targets != -100).any(), "A long prompt must still train on its response."


def test_finetune_checkpoint_is_reloadable(tmp_path):
    from scripts.finetune_instruct import _save_checkpoint
    from models.custom_minilm.checkpoint import load_model_from_checkpoint
    from models.custom_minilm.config import MiniLLMConfig
    from models.custom_minilm.model import MiniLLM

    config = MiniLLMConfig(vocab_size=32, d_model=16, n_layers=1, n_heads=2, d_ff=32)
    model = MiniLLM(config)
    optimizer = torch.optim.AdamW(model.parameters())
    checkpoint = tmp_path / "finetune.pt"
    _save_checkpoint(
        model, optimizer, step=3, loss=1.0, path=str(checkpoint),
        tokenizer_path="tokenizer/vocab/demo_bpe_vocab.json",
    )

    restored, meta = load_model_from_checkpoint(checkpoint)
    assert restored.config == config
    assert meta["tokenizer_path"] == "tokenizer/vocab/demo_bpe_vocab.json"


def test_rag_dimension_mismatch_is_skipped():
    from rag.chunking import TextChunk
    from rag.index import LocalVectorIndex

    index = LocalVectorIndex()
    index.add(TextChunk("ok", "test", "matching"), [1.0, 0.0])
    index.add(TextChunk("bad", "test", "mismatched"), [1.0, 0.0, 0.0])

    results = index.search([1.0, 0.0])
    assert [result["chunk_id"] for result in results] == ["ok"]
