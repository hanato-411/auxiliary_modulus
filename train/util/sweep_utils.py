from typing import Any, Dict, Iterable, List, Tuple

import wandb
from omegaconf import OmegaConf, DictConfig


def _undot_keys(flat_dict: Dict[str, Any]) -> Dict[str, Any]:
    """dot.notation キーをネスト辞書に戻す。"""
    nested: Dict[str, Any] = {}
    for key, value in flat_dict.items():
        if not isinstance(key, str):
            nested[key] = value
            continue

        parts = key.split(".")
        target = nested
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value
    return nested


def _merge_wandb_overrides(cfg: DictConfig) -> DictConfig:
    """wandb.config の sweep override を OmegaConf にマージする。"""
    if not wandb.config:
        return cfg

    filtered = {
        key: value
        for key, value in wandb.config.as_dict().items()
        if not str(key).startswith("_")
    }
    if not filtered:
        return cfg

    sweep_cfg = OmegaConf.create(_undot_keys(filtered))
    return OmegaConf.merge(cfg, sweep_cfg)


def _format_metadata_value(value: Any) -> str:
    """タグ/ラン名向けに値を文字列フォーマットする。"""
    if isinstance(value, float):
        if value == 0:
            return "0"
        if abs(value) >= 1e-2:
            return f"{value:.4f}".rstrip("0").rstrip(".")
        return f"{value:.0e}"
    return str(value)


def _collect_metadata_pairs(cfg: DictConfig, keys: Iterable[str]) -> List[Tuple[str, Any]]:
    pairs: List[Tuple[str, Any]] = []
    for path in keys:
        if not path:
            continue
        value = OmegaConf.select(cfg, path, default=None)
        if value is None:
            continue
        key_name = path.split(".")[-1]
        pairs.append((key_name, value))
    return pairs


def _normalize_dynamic_keys(dynamic_keys: Any) -> List[str]:
    """
    wandb.dynamic_metadata_keys を「ドット区切りパスの文字列のリスト」に正規化する。
    辞書が渡された場合（マージ結果で意図せず全体が入った場合）は空リストを返し、
    save_dir がフル設定文字列になるのを防ぐ。
    """
    if dynamic_keys is None:
        return []
    if isinstance(dynamic_keys, dict):
        # 辞書のままイテレするとトップレベルキー(add/train/model等)になり、
        # 各 value が辞書で suffix がフル設定文字列になって File name too long の原因になる
        return []
    try:
        keys_list = list(dynamic_keys)
    except TypeError:
        return []
    result = []
    for k in keys_list:
        if isinstance(k, str) and k.strip():
            result.append(k.strip())
    return result


def _format_ratio_for_name(value: Any) -> str:
    """run_name/save_dir 向けに ratio を安全な短い文字列へ変換する。"""
    try:
        ratio = float(value)
    except (TypeError, ValueError):
        return str(value)
    # 0.3 -> 0p3, 0.125 -> 0p125
    return f"{ratio:g}".replace(".", "p")


def _normalize_train_size(value: Any) -> str:
    """
    train_size を "1M"/"100K"/"10K" のような表記に正規化する。
    sweep 側で int/str のどちらを渡しても扱えるようにする。
    """
    if value is None:
        return "1M"
    if isinstance(value, int):
        if value >= 1_000_000 and value % 1_000_000 == 0:
            return f"{value // 1_000_000}M"
        if value >= 1_000 and value % 1_000 == 0:
            return f"{value // 1_000}K"
        return str(value)

    text = str(value).strip().upper()
    aliases = {
        "1000000": "1M",
        "100000": "100K",
        "10000": "10K",
    }
    return aliases.get(text, text)


def _apply_sweep_dynamic_fields(cfg: DictConfig) -> None:
    """
    q/N/K/ratio/train_size から
    save_dir, dataset_pair_spec, run_name, tags を自動生成する。
    """
    enabled = bool(OmegaConf.select(cfg, "train.dynamic_from_sweep", default="True"))
    if not enabled:
        return

    q = int(OmegaConf.select(cfg, "train.q"))
    N = int(OmegaConf.select(cfg, "train.N"))
    k = OmegaConf.select(cfg, "train.K")
    ratio = OmegaConf.select(cfg, "train.ratio")
    ratio_name = _format_ratio_for_name(ratio)
    train_size = _normalize_train_size(OmegaConf.select(cfg, "train.train_size", default="1M"))
    embed_type = str(OmegaConf.select(cfg, "model.embed_type", default="angular"))
    weight_init = str(OmegaConf.select(cfg, "model.weight_init", default="True"))
    bias = str(OmegaConf.select(cfg, "model.bias", default="True"))
    dropout = str(OmegaConf.select(cfg, "model.dropout", default="0.0"))
    norm_first = str(OmegaConf.select(cfg, "model.norm_first", default="True"))

    mode = str(OmegaConf.select(cfg, "train.mode", default="normal"))
    if mode == "normal":
        train_kind = "default"
        test_kind = "default"
    elif mode == "mod_Kq":
        train_kind = "add_Kq"
        test_kind = "add_Kq"
    elif mode == "sparse":
        train_kind = "inv_sqrt"
        test_kind = "default"
    else:
        raise ValueError(f"Invalid mode: {mode}")

    train_dataset_root = f"data/N{N}_q{q}/{train_kind}/train{train_size}_test10K"
    test_dataset_root = f"data/N{N}_q{q}/{test_kind}/train{train_size}_test10K"
    train_dataset_path = f"{train_dataset_root}/train_raw.txt"
    test_dataset_path = f"{test_dataset_root}/test_raw.txt"

    cfg["data"]["train_dataset_path"] = train_dataset_path
    cfg["data"]["test_dataset_path"] = test_dataset_path

    if train_kind == "add_Kq":
        run_name = f"mod-Q{q}-N{N}-mod_Kq-K{k}-ratio{ratio_name}-train{train_size}-{embed_type}"
    elif train_kind == "inv_sqrt":
        run_name = f"mod-Q{q}-N{N}-sparse-train{train_size}-{embed_type}"
    elif train_kind == "default":
        run_name = f"mod-Q{q}-N{N}-default-train{train_size}-{embed_type}"
    else:
        raise ValueError(f"Invalid train_kind: {train_kind}")
    cfg["wandb"]["run_name"] = run_name

    if cfg["train"]["save_dir"] is None:
        if train_kind == "add_Kq":
            cfg["train"]["save_dir"] = f"results/{embed_type}/{mode}/N={N}/q={q}/K={k}/r={ratio_name}"
        else:
            cfg["train"]["save_dir"] = f"results/{embed_type}/{mode}/N={N}/q={q}"

    existing_tags = list(cfg["wandb"].get("tags", []))
    dynamic_tags = [
        f"q:{q}",
        f"N:{N}",
        f"K:{k}",
        f"ratio:{ratio}",
        f"train_size:{train_size}",
        f"mode:{mode}",
        f"embed_type:{embed_type}",

    ]
    cfg["wandb"]["tags"] = existing_tags + dynamic_tags

    if wandb.run is not None:
        wandb.run.name = run_name
        wandb.run.tags = tuple(cfg["wandb"]["tags"])

