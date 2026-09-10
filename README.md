# SovereignAI — Self-Hosted, Air-Gapped AI Workbench

**Smart India Hackathon 2026 — Problem Statement SIH 117**

Built by: **Akash Upadhyay** (B.Tech CS / AI-ML)

> **Honesty first.** The "custom LLM" in this project is a genuine,
> from-scratch, randomly-initialized, self-trained **educational /
> research MiniLLM** — not a claim of frontier-model performance. See
> [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §2 for an exact,
> unembellished account of what is custom, what is open-weight, and
> what is rule-based. `docs/FINAL_AUDIT.md` will make this final and
> complete once every phase is done.

## What this project is

A self-hosted AI workbench designed to run entirely on an
organization's own hardware, with **zero runtime dependency on any
cloud AI API**, so that confidential engineering / PSU / defence-
adjacent documents never leave the premises. Full system design:
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Build status

This project is built in **24 phase-gated milestones** so every piece
is genuinely understood, implemented, and tested before the next one
is added. Nothing is faked to look finished. Live status:
[`docs/BUILD_STATUS.md`](docs/BUILD_STATUS.md).

Currently complete: **Phase 0 (architecture) → Phase 3 (tokenizer)**.

## Quick start (Windows 11)

```powershell
git clone <this-repo-url>
cd sovereign-ai-workbench
.\scripts\setup_windows.ps1
```

This creates a virtual environment, installs dependencies, and runs
`scripts/verify_gpu.py` so you can confirm your machine (e.g. an RTX
4050 laptop GPU, 6 GB VRAM) is visible to PyTorch.

## Try what already works (Phases 0-3)

```powershell
python scripts\prepare_data.py         # build data/cleaned + data/processed from data/raw
python scripts\train_tokenizer.py      # train the from-scratch BPE tokenizer
python -m pytest tokenizer\tests -v    # run the tokenizer test suite
```

All three of these were also run and verified inside the build
sandbox that produced this repository — see the phase reports in
`docs/BUILD_STATUS.md` for the exact output.

## Project structure & roadmap

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full,
annotated folder structure, dependency plan, and the 24-phase roadmap.
Folders for phases not yet built contain only a short `README.md`
explaining what will go there — no placeholder/fake code.

## License

MIT — see [`LICENSE`](LICENSE).
