"""使用GPUの限定（CUDA_VISIBLE_DEVICES）。torch / transformers より前に apply_visible_gpus_before_torch を呼ぶこと。"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Sequence, Tuple


def _gpu_ids_to_env_string(gpu_ids: Any) -> Optional[str]:
    if gpu_ids is None:
        return None
    if isinstance(gpu_ids, (list, tuple)):
        parts = [str(int(x)).strip() for x in gpu_ids]
        return ",".join(parts) if parts else None
    s = str(gpu_ids).strip()
    return s if s else None


def apply_visible_gpus_before_torch(
    config_path: Path,
    *,
    env_gpu_ids_names: Sequence[str] = ("TRAIN_GPU_IDS",),
    yaml_gpu_sources: Tuple[Tuple[str, str], ...] = (("train", "gpu_ids"),),
) -> None:
    """
    使用する物理GPU番号を指定する（PyTorch から見えるデバイスは常に 0 から連番になる）。

    優先順位:
    1. 既に CUDA_VISIBLE_DEVICES が設定されている → 変更しない
    2. env_gpu_ids_names に列挙した環境変数のうち、最初に空でないもの（例: TRAIN_GPU_IDS）
    3. yaml_gpu_sources の順に (セクション名, キー) の gpu_ids を読む。ファイルが無い・全て空はスキップ

    訓練スクリプトのデフォルト: env は TRAIN_GPU_IDS のみ、YAML は train.gpu_ids。
    データセット生成: apply_visible_gpus_for_dataset() を使う。
    """
    if os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return
    for env_name in env_gpu_ids_names:
        env_spec = os.environ.get(env_name, "").strip()
        if env_spec:
            os.environ["CUDA_VISIBLE_DEVICES"] = env_spec.replace(" ", "")
            return
    if not config_path.is_file():
        return
    try:
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return

    cfg = OmegaConf.load(config_path)
    for section_name, key in yaml_gpu_sources:
        section = cfg.get(section_name) or {}
        spec = _gpu_ids_to_env_string(section.get(key))
        if spec:
            os.environ["CUDA_VISIBLE_DEVICES"] = spec
            return


def apply_visible_gpus_for_dataset(config_path: Path) -> None:
    """
    データセット生成用。優先順位:
    1. CUDA_VISIBLE_DEVICES（既存）
    2. DATASET_GPU_IDS
    3. TRAIN_GPU_IDS
    4. add.gpu_ids → train.gpu_ids（YAML）
    """
    apply_visible_gpus_before_torch(
        config_path,
        env_gpu_ids_names=("DATASET_GPU_IDS", "TRAIN_GPU_IDS"),
        yaml_gpu_sources=(("add", "gpu_ids"), ("train", "gpu_ids")),
    )


def scaled_per_device_batch_size(global_bs: int) -> int:
    """従来どおり、config のバッチを可见 CUDA デバイス数で割る。CPU または CUDA 非利用時は割らない。"""
    import torch

    if not torch.cuda.is_available():
        return int(global_bs)
    n = int(torch.cuda.device_count())
    if n <= 0:
        return int(global_bs)
    return max(1, int(global_bs) // n)
