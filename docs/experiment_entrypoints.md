# Experiment Entrypoints

This document lists practical entrypoints for dataset generation and training.

## Prerequisites

```bash
make build
make run
```

## 1. Dataset Generation

Generate datasets with:

```bash
python3 -m scripts.generate_dataset --config config/generate/group_a_small.yaml
python3 -m scripts.generate_dataset --config config/generate/group_b_mid.yaml
python3 -m scripts.generate_dataset --config config/generate/group_c_large.yaml
```

## 2. Single Training Run

Run training inside the container:

```bash
docker exec -it auxiliary_modular_addition-container bash -lc \
  'cd /app && sage train/train.py'
```

All runtime settings are loaded from `config/config_train.yaml`.

## 3. Sweep Experiments

Create a sweep config from:

- `config/sweep/templates/template_sweep.yaml`

Then run:

```bash
wandb sweep <your-sweep-config.yaml>
wandb agent <entity/project/sweep_id>
```

## 4. Related Docs

- `docs/dataset_generation_settings.md`
- `docs/train_mod_settings.md`
- `docs/sweep_parameter_guide.md`
