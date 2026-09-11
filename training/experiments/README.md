# Sovereign AI Workbench — Training Experiments

This directory stores experiment configuration manifests, hyperparameter definitions, and run logs for reproducible model training.

## Experiment Profiles

| Experiment | Model Family | Parameters | Context Length | d_model | Layers | Heads | FFN Dim | Primary Focus |
|------------|-------------|------------|----------------|---------|--------|-------|---------|---------------|
| `exp1_10m` | Custom Transformer | ~10.2M | 512 | 256 | 8 | 8 | 1024 | Primary sovereign LLM configuration |
| `exp2_13m` | Custom Transformer | ~13.5M | 512 | 384 | 8 | 6 | 1536 | Experimental capacity & attention scaling |

## Reproducibility Workflow
To train with an experiment configuration:
```powershell
python scripts/train_model.py --experiment exp1 --train-file data/processed/openorca_pretrain.txt --vocab-path tokenizer/vocab/bpe_8k.json
```
