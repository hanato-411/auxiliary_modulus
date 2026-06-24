# `train.py` Configuration Guide

This document explains the configuration fields in `config/config_train.yaml`, which is loaded by `train/train.py`.

- Single run: edit `config/config_train.yaml` and run training directly.
- Multi-condition comparison: create sweep configs and run `wandb sweep` / `wandb agent`.

## 1. `train.mode` (training mode)

`train.mode` is the main behavior switch in this project.
Allowed values are `normal`, `sparse`, and `mod_Kq`.

- `normal`
  - Standard training mode.
  - Typically used with `default` datasets.
- `sparse`
  - Sparse-learning mode.
  - Typically used with `inv_sqrt` datasets.
- `mod_Kq`
  - Mode for auxiliary modulus training.
  - Typically used with `add_Kq` datasets.
  - `train.K` and `train.ratio` are important in this mode.

## 2. `train` section

### Basic hyperparameters

- `train.N`: input sequence length
- `train.q`: modulus
- `train.K`: coefficient parameter for `mode=mod_Kq`
- `train.ratio`: ratio parameter for `mode=mod_Kq`
- `train.choice_n`: number of candidates for `mod_Kq` mode
- `train.seed`: random seed
- `train.gpu_ids`: GPU selection when `CUDA_VISIBLE_DEVICES` is not set

### Optimization and schedule

- `train.num_train_epochs`: number of epochs
- `train.learning_rate`: learning rate
- `train.optimizer`: optimizer type
- `train.per_device_train_batch_size`: train batch size
- `train.per_device_eval_batch_size`: eval batch size
- `train.lr_scheduler_type`: scheduler type
- `train.warmup_ratio`: warmup ratio
- `train.weight_decay`: weight decay
- `train.init_std`: initialization scale

### Logging and checkpoints

- `train.save_strategy`: checkpoint strategy (for example, `steps`)
- `train.save_steps`: checkpoint interval
- `train.logging_steps`: logging interval
- `train.eval_steps`: evaluation interval
- `train.save_dir`: output path (`null` enables auto-generation)

### Sweep helpers

- `train.dynamic_from_sweep`: dynamically derive paths/names from sweep values
- `train.train_size`: training-size label used in sweeps (for example, `100K`, `1M`)
- `train.token_tolerance_ratio_list` (optional): token-eval threshold ratios

## 3. `model` section

### Architecture

- `model.hidden_size`
- `model.num_encoder_layers`
- `model.num_decoder_layers` (kept for compatibility)
- `model.num_attention_heads`
- `model.dim_feedforward`
- `model.dropout`
- `model.activation`
- `model.layer_norm_eps`
- `model.norm_first`
- `model.bias`
- `model.positional_embedding`
- `model.embed_type` (`token` or `angular`)
- `model.type` (typically `encoder_only`)

### Initialization

- `model.weight_init`
- `model.linear_init_type` (default: `normal`)
- `model.embedding_init_type` (default: `normal`)

## 4. `wandb` section

- `wandb.project_name`: W&B project name
- `wandb.run_name`: run name (may be overwritten by sweep logic)
- `wandb.entity`: W&B entity (optional)
- `wandb.tags`: list of tags
- `wandb.log_model`: model logging setting (optional)

## 5. `data` section

- `data.train_dataset_path`: training dataset path (may be overwritten by sweep logic)
- `data.test_dataset_path`: evaluation dataset path (may be overwritten by sweep logic)
- `data.num_train_samples`: max training samples (`-1` means all)
- `data.num_test_samples`: max evaluation samples (`-1` means all)

Currently unused fields:

- `data.lexer_config`
- `data.validate_train_tokens`
- `data.validate_test_tokens`
- `data.display_samples`

## 6. `save_dir` auto-generation (`save_dir: null`)

When `train.dynamic_from_sweep: true` and `train.save_dir: null`, `save_dir` is generated as:

- `normal` / `sparse`:
  - `results/{model.embed_type}/{train.mode}/{train.N}/{train.q}`
- `mod_Kq`:
  - `results/{model.embed_type}/{train.mode}/{train.N}/{train.q}/{train.K}_{train.ratio}`

## 7. Typical operation

- Single run:
  - Edit `config/config_train.yaml`
  - Run `sage train/train.py`
- Sweep runs:
  - Copy `config/sweep/templates/template_sweep.yaml`
  - Run `wandb sweep` / `wandb agent` with your sweep config
