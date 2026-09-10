"""
scripts/run_inference.py — Generate text with a trained MiniLLM.

Usage:
    python scripts\\run_inference.py --checkpoint checkpoints/exp2_step00300_final.pt --prompt "The pump inspection"
    python scripts\\run_inference.py --checkpoint checkpoints/exp2_step00300_final.pt --prompt "The inspection report" --temperature 0.8 --top-k 40
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tokenizer.tokenizer import ByteLevelBPETokenizer
from models.custom_minilm.checkpoint import load_model_from_checkpoint
from models.custom_minilm.generate import generate


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference with trained MiniLLM")
    p.add_argument("--checkpoint", required=True, help="Path to .pt checkpoint file")
    p.add_argument("--prompt", default="The inspection", help="Text prompt")
    p.add_argument("--max-new-tokens", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=40)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    # Load model
    model, meta = load_model_from_checkpoint(args.checkpoint, device=device)
    print(f"Loaded: step={meta['step']}  params={meta['model_params']:,}")

    # Load tokenizer
    tok_path = meta.get("tokenizer_path") or "tokenizer/vocab/demo_bpe_vocab.json"
    tokenizer = ByteLevelBPETokenizer.load(ROOT / tok_path)

    # Encode prompt
    prompt_ids = tokenizer.encode(args.prompt, add_bos=True)

    print(f"\nPrompt : {args.prompt!r}")
    print(f"Tokens : {len(prompt_ids)}")
    print("─" * 50)

    # Generate
    eos_id = tokenizer.vocab.special_to_id.get("<eos>")
    new_ids = generate(
        model,
        prompt_ids=prompt_ids,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        eos_id=eos_id,
        seed=args.seed,
        device=device,
    )

    full_ids = prompt_ids + new_ids
    output_text = tokenizer.decode(full_ids)
    print(output_text)
    print("─" * 50)
    print(f"Generated {len(new_ids)} new tokens.")


if __name__ == "__main__":
    main()
