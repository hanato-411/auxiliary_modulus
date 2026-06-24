# data_sample

Place dataset files for `train/train.py` in this directory.

- `train_raw.txt`
- `test_raw.txt`

You can generate datasets with:

```bash
python3 scripts/generate_dataset.py --config config/config_generate.yaml
```

Single-run CLI example:

```bash
python3 scripts/generate_dataset.py --q 257 --N 128 --train-size 100000 --test-size 10000 --type add_Kq --output-dir data_sample
```

Output format:

`a | b | ... # y | k`

- Left side: input sequence (length `N`)
- `y`: `(sum x_i) mod q`
- `k`: `floor((sum x_i) / q)`
