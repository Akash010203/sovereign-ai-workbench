"""
Phase 1 acceptance check.

Run on the TARGET machine (Windows 11 + RTX 4050 laptop GPU, or any
machine with Python + PyTorch installed):

    python scripts/verify_gpu.py

This script does not train or load any model. It only reports what
hardware/software is available, so the project has a written record of
what it actually ran on -- instead of just claiming "it runs on GPU".
"""
from __future__ import annotations

import platform
import sys


def main() -> None:
    print("=" * 60)
    print("SovereignAI - Environment / GPU verification")
    print("=" * 60)
    print(f"Python version : {sys.version.split()[0]}")
    print(f"Platform       : {platform.platform()}")

    try:
        import torch
    except ImportError:
        print("PyTorch        : NOT INSTALLED")
        print()
        print("Install it with (see requirements.txt):")
        print("    python -m pip install torch")
        return

    print(f"PyTorch version: {torch.__version__}")
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available : {cuda_available}")

    if cuda_available:
        device_index = 0
        name = torch.cuda.get_device_name(device_index)
        total_vram_gb = (
            torch.cuda.get_device_properties(device_index).total_memory
            / (1024**3)
        )
        print(f"GPU name       : {name}")
        print(f"Total VRAM     : {total_vram_gb:.2f} GB")
        print()
        print("Result: training scripts will use CUDA.")
    else:
        print()
        print("Result: no CUDA GPU detected -> CPU fallback will be used.")
        print("Training still works, but is limited to very small")
        print("smoke-test experiments without a GPU.")


if __name__ == "__main__":
    main()
