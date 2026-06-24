#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import random
from pathlib import Path
from typing import Any, List, Tuple
import train.util.datetime_utc_compat
from train.gpu_setup import apply_visible_gpus_for_dataset

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate modular-addition datasets via calt DatasetGenerator."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config/config_generate.yaml"),
        help="dataset generation config (private config add-section compatible)",
    )
    parser.add_argument("--q", type=int, help="modulus q (single-run mode)")
    parser.add_argument("--N", type=int, help="sequence length N (single-run mode)")
    parser.add_argument("--train-size", type=int, default=100000)
    parser.add_argument("--test-size", type=int, default=10000)
    parser.add_argument(
        "--type",
        choices=["add_Kq", "inv_sqrt", "default"],
        default="add_Kq",
        help="dataset type for single-run mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data_sample"),
        help="output directory for single-run mode",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--backend", type=str, default="multiprocessing")
    return parser.parse_args()


class MODAddProblemGenerator:
    """Generator compatible with calt.dataset.sagemath.DatasetGenerator."""

    def __init__(self, q: int, N: int, dataset_type: str = "add_Kq"):
        self.q = int(q)
        self.N = int(N)
        self.dataset_type = str(dataset_type)

    def __call__(self, seed: int) -> Tuple[List[int], Any]:
        import sage.misc.randstate as randstate  # type: ignore[reportMissingImports]

        randstate.set_random_seed(seed)
        random.seed(seed)

        if self.dataset_type == "add_Kq":
            X = self.default()
            quotient, remainder = divmod(sum(X), self.q)
            return X, [remainder, quotient]

        if self.dataset_type == "inv_sqrt":
            X = self.inv_sqrt()
            return X, sum(X) % self.q

        if self.dataset_type == "default":
            X = self.default()
            return X, sum(X) % self.q

        raise ValueError(f"Unsupported type: {self.dataset_type}")

    def default(self) -> List[int]:
        import sage.misc.prandom as prandom  # type: ignore[reportMissingImports]

        return [prandom.randint(0, self.q - 1) for _ in range(self.N)]

    def inv_sqrt(self) -> List[int]:
        probs = [1 / math.sqrt(self.N - z + 1) for z in range(1, self.N + 1)]
        total = sum(probs)
        probs = [p / total for p in probs]
        z = random.choices(range(1, self.N + 1), weights=probs)

        non_zero_positions = set(random.sample(range(self.N), z[0]))
        values = [0] * self.N
        for i in non_zero_positions:
            while True:
                val = random.randint(1, self.q - 1)
                if val != 0:
                    values[i] = val
                    break
        return values


def _resolve_qN_pairs_and_types(cfg_add) -> Tuple[List[Tuple[int, int]], List[str]]:
    """
    Return (q, N) pairs and dataset types.
    Priority:
    1) q_list + N_list (cartesian product)
    2) qN_pairs (explicit pairs)
    3) single q + N + type
    """
    types_list = list(cfg_add.get("dataset_types") or ["default", "add_Kq", "inv_sqrt"])

    q_list = cfg_add.get("q_list")
    N_list = cfg_add.get("N_list")
    if q_list is not None and N_list is not None:
        ql = [int(x) for x in q_list]
        nl = [int(x) for x in N_list]
        if ql and nl:
            return [(q, n) for q in ql for n in nl], types_list

    qn_pairs_cfg = cfg_add.get("qN_pairs")
    if qn_pairs_cfg is not None:
        pairs = [(int(item.q), int(item.N)) for item in qn_pairs_cfg]
        if pairs:
            return pairs, types_list

    q = int(cfg_add["q"])
    n = int(cfg_add["N"])
    typ = str(cfg_add.get("type", "add_Kq"))
    return [(q, n)], [typ]


def _resolve_sample_size_pairs(cfg_add) -> List[Tuple[int, int]]:
    """
    Return (train_size, test_size) list.
    Priority:
    1) sample_size_pairs
    2) train_sample_size + test_sample_size
    """
    size_pairs_cfg = cfg_add.get("sample_size_pairs")
    if size_pairs_cfg is not None:
        pairs: List[Tuple[int, int]] = []
        for item in size_pairs_cfg:
            pairs.append((int(item.train), int(item.test)))
        if pairs:
            return pairs

    train_size = int(cfg_add["train_sample_size"])
    test_size = int(cfg_add["test_sample_size"])
    return [(train_size, test_size)]


def _format_sample_count_short(n: int) -> str:
    units = [("B", 1_000_000_000), ("M", 1_000_000), ("K", 1_000)]
    for suffix, base in units:
        if n >= base:
            if n % base == 0:
                return f"{n // base}{suffix}"
            value = n / base
            text = f"{value:.2f}".rstrip("0").rstrip(".")
            return f"{text}{suffix}"
    return str(n)


def run_generation(
    *,
    q: int,
    n: int,
    dataset_type: str,
    train_size: int,
    test_size: int,
    output_dir: Path,
    seed: int,
    backend: str,
    n_jobs: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    from calt.dataset.sagemath import DatasetGenerator  # type: ignore[reportMissingImports]

    problem_generator = MODAddProblemGenerator(
        q=q,
        N=n,
        dataset_type=dataset_type,
    )
    dataset_generator = DatasetGenerator(
        backend=backend,
        n_jobs=n_jobs,
        verbose=True,
        root_seed=seed,
    )
    dataset_generator.run(
        dataset_sizes={"train": train_size, "test": test_size},
        instance_generator=problem_generator,
        save_dir=str(output_dir),
    )


def run_from_config(config_path: Path) -> None:
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "omegaconf is required for --config mode. "
            "Run inside Docker (`make run`) or install dependencies first."
        ) from exc
    
    cfg = OmegaConf.load(config_path)
    add_cfg = cfg["add"]

    random_seed = int(add_cfg.get("random_seed", 42))
    save_dir_template = str(
        add_cfg.get("save_dir_template") or "data/N{N}_q{q}/{type}"
    )
    backend = str(add_cfg.get("backend", "multiprocessing"))
    n_jobs = int(add_cfg.get("n_jobs", -1))

    qn_pairs, dataset_types = _resolve_qN_pairs_and_types(add_cfg)
    sample_size_pairs = _resolve_sample_size_pairs(add_cfg)

    for q, n in qn_pairs:
        for dataset_type in dataset_types:
            for train_size, test_size in sample_size_pairs:
                base_dir = save_dir_template.format(N=n, q=q, type=dataset_type)
                train_short = _format_sample_count_short(train_size)
                test_short = _format_sample_count_short(test_size)
                output_dir = Path(f"{base_dir}/train{train_short}_test{test_short}")

                print(
                    "=== Generating "
                    f"q={q} N={n} type={dataset_type} "
                    f"train={train_size} test={test_size} -> {output_dir} ==="
                )
                run_generation(
                    q=q,
                    n=n,
                    dataset_type=dataset_type,
                    train_size=train_size,
                    test_size=test_size,
                    output_dir=output_dir,
                    seed=random_seed,
                    backend=backend,
                    n_jobs=n_jobs,
                )


def main() -> None:
    args = parse_args()
    if args.q is not None or args.N is not None:
        if args.q is None or args.N is None:
            raise ValueError("single-run mode requires both --q and --N")
        if args.q < 2:
            raise ValueError("q must be >= 2")
        if args.N < 1:
            raise ValueError("N must be >= 1")

        run_generation(
            q=args.q,
            n=args.N,
            dataset_type=args.type,
            train_size=args.train_size,
            test_size=args.test_size,
            output_dir=args.output_dir,
            seed=args.seed,
            backend=args.backend,
            n_jobs=args.n_jobs,
        )
        print(f"generated by calt at: {args.output_dir}")
        return

    run_from_config(args.config)


if __name__ == "__main__":
    main()
