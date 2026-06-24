# Sweep Parameter Guide

This guide describes how to tune sweep parameters for reproducible experiments.

## Recommended Workflow

1. Start from `config/sweep/templates/template_sweep.yaml`.
2. Edit parameter axes for your experiment.
3. Run a sweep.

```bash
wandb sweep <your-sweep-config.yaml>
wandb agent <entity/project/sweep_id>
```

## Main Parameters

- `train.q`  
  Modulus `q` used by the task.

- `train.seq_len`  
  Input length `N` (if your config uses this key).

- `train.N`
  Input length `N` (used in current training config).

- `train.train_size`  
  Training dataset size label, for example `"100K"`, `"1M"`, `"10M"`.

- `train.dataset_kind_pair`  
  Dataset variant mapping for train/test.

- `model.embed_type`  
  Embedding type (`token` or `angular`).

- `train.loss_target_mode`  
  Training objective mode, typically `mod_Kq`, `normal`, or `sparse`.

- `train.K`  
  Auxiliary modulus coefficient for `mod_Kq`.

- `train.choice_shifted_ratio`  
  Shift ratio parameter `r`.

- `model.norm_first`, `model.bias`, `model.dropout`, `model.weight_init`
  Useful axes for architecture ablations.

## Notes

- Keep sweep configs near `config/sweep/` for consistency.
- Reproducibility-focused sweep configs are planned under `sweep/experiments`.
- Use descriptive run names and tags to simplify W&B comparisons.
