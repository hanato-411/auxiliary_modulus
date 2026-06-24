# Dataset Generation Settings

This document describes the dataset-generation configuration used by `scripts/generate_dataset.py`.

## 1. Where to configure

Dataset generation is configured in YAML files under `config/generate/`.

Common files:
- `config/generate/group_a_small.yaml`
- `config/generate/group_b_mid.yaml`
- `config/generate/group_c_large.yaml`
- `config/generate/config_generate.yaml` (general template)

All files share the same top-level section: `add`.

## 2. `(q, N)` selection priority

The script resolves `(q, N)` combinations in the following order:

1. `q_list` + `N_list`
   - Uses the Cartesian product of all values.
2. `qN_pairs`
   - Uses explicit `(q, N)` pairs.
3. legacy single values
   - Falls back to `q`, `N`, `type` when list/pair settings are not provided.

## 3. Key parameters

- `q_list`
  - Candidate modulus values.
- `N_list`
  - Candidate sequence lengths.
- `qN_pairs`
  - Explicit pair list for non-Cartesian combinations only.
- `dataset_types`
  - Dataset variants to generate.
  - `default`: uniformly random samples.
  - `inv_sqrt`: sparse dataset generation (baseline in the paper).
  - `add_Kq`: uniformly random samples plus quotient information in labels (used by the proposed method).
- `sample_size_pairs`
  - List of `{train, test}` pairs.
  - Each pair generates one dataset-size variant.
- `gpu_ids`
  - Optional GPU visibility hint for dataset generation helpers.
- `save_dir_template`
  - Output base path template.
  - Supported placeholders: `{N}`, `{q}`, `{type}`.
- `random_seed`
  - Root seed for reproducible generation.
- `backend`
  - Backend passed to `calt.dataset.sagemath.DatasetGenerator` (for example, `multiprocessing`).
- `n_jobs`
  - Number of workers (`-1` means auto/max).

## 4. Output directory format

The script appends a size suffix to `save_dir_template`:

- `.../train{train_size_short}_test{test_size_short}`

Examples:
- `train100K_test10K`
- `train1M_test10K`
- `train10M_test10K`

## 5. Typical command

```bash
python3 -m scripts.generate_dataset --config config/generate/group_a_small.yaml
```
