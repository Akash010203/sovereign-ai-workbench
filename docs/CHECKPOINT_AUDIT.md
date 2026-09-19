# SovereignAI — Checkpoint Audit

> Generated: 2026-09-19 by forensic audit.

## Summary

| Series | Count | Total Size | Architecture | Best Checkpoint |
|---|---|---|---|---|
| exp1_smoke | 5 | ~280 MB | Small (~5M params, d_model=256) | `exp1_smoke_step00050_best.pt` |
| exp4_openorca | ~280 | ~50 GB | Medium (~12M params, d_model=384) | `exp4_openorca_step30000_best.pt` |
| industrial_80m | ~42 | ~40 GB | Industrial 80M (~83M, d_model=768) | `industrial_80m_best.pt` |
| standalone | 3 | ~2.1 GB | Various | `best.pt`, `industrial_80m_best.pt`, `industrial_80m_smoke.pt` |
| finetuned | dir | unknown | Fine-tuned variants | `finetuned/finetune_best.pt` |
| audit | 2 | ~115 MB | Various | `_audit_pretrain.pt`, `_audit_finetune/` |

**Total: ~452 files, ~117 GB**

## File Size Patterns

| Size ~56 MB | Size ~197 MB | Size ~207 MB | Size ~951 MB |
|---|---|---|---|
| Small model (5M params) | Medium periodic save | Medium best checkpoint | Industrial 80M (83M params) |
| `exp1_smoke_*` | `exp4_openorca_step*XXXX.pt` | `exp4_openorca_step*_best.pt` | `industrial_80m_*` |

### Size Discrepancy in exp4_openorca

Some `exp4_openorca_*_best.pt` files are ~56 MB instead of ~207 MB. This suggests these checkpoints may have been saved with a different model configuration (small instead of medium), or only model weights without optimizer state. This warrants investigation.

## Active Checkpoints (Currently Used by Backend)

The backend loads checkpoints in this priority order:
1. `checkpoints/industrial_80m_best.pt` (951 MB) — **ACTIVE** ✅
2. `checkpoints/best_8k.pt` — not present
3. `checkpoints/finetuned/finetune_best.pt` — may exist
4. `checkpoints/best.pt` (207 MB) — fallback

## Training Log Summary

### industrial_80m_4050 (Best Available)

| Metric | Value |
|---|---|
| Total steps | 20,000 |
| Starting train loss | 9.816 |
| Final train loss | 2.375 |
| Final val loss | 3.570 |
| Final val perplexity | 35.5 |
| Starting gradient norm | 10.94 |
| Final gradient norm | 1.75 |
| Step duration | ~0.9s |

### Recommendations

1. **Prune intermediate checkpoints**: Keep only `*_best.pt`, `*_final.pt`, and milestone steps (every 5000). This could recover ~60 GB of disk space.
2. **Investigate size-mismatched best checkpoints**: Some `exp4_openorca_*_best.pt` files are 56 MB instead of 207 MB.
3. **Verify checkpoint compatibility**: Ensure `industrial_80m_best.pt` loads with the `blended_16k.json` tokenizer.
