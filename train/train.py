import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import util.datetime_utc_compat
from omegaconf import OmegaConf, DictConfig
from transformers import TrainingArguments, Trainer as HFTrainer
import wandb
import torch
import numpy as np
from calt import Trainer
from typing import Dict, Any, List
from torch.utils.data import Dataset
from tqdm import tqdm

from gpu_setup import apply_visible_gpus_before_torch, scaled_per_device_batch_size
apply_visible_gpus_before_torch(PROJECT_ROOT / "config" / "config_train.yaml")

from util.sweep_utils import (
    _merge_wandb_overrides,
    _apply_sweep_dynamic_fields,
)

from hf_transformer_encoder_embed import (  # noqa: E402
    EncoderOnlyTransformerConfig,
    TransformerForEncoderOnly,
)

class TrainerNoAutoEvalSave(Trainer):
    """`evaluate` 時の自動 generation 保存を無効化する。"""

    def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
        return HFTrainer.evaluate(
            self,
            eval_dataset=eval_dataset,
            ignore_keys=ignore_keys,
            metric_key_prefix=metric_key_prefix,
        )


class RawIdDataset(Dataset):
    """`a | b | ... # y | k` 形式を数値IDとして直接読む Dataset。"""

    def __init__(
        self,
        data_path: str,
        q: int,
        dummy_token_id: int = 0,
        max_samples: int | None = None,
        show_progress: bool = True,
        progress_desc: str | None = None,
    ):
        self.samples: List[Dict[str, Any]] = []
        self.max_token_id = 0
        self.q = int(q)
        self.dummy_token_id = int(dummy_token_id)
        self.show_progress = bool(show_progress)
        if max_samples == -1:
            max_samples = None
        total_lines = None
        if self.show_progress:
            with open(data_path, "r", encoding="utf-8") as f_count:
                total_lines = sum(1 for _ in f_count)

        desc = progress_desc or f"RawIdDataset {Path(data_path).name}"
        with open(data_path, "r", encoding="utf-8") as f:
            line_iter = f
            if self.show_progress:
                line_iter = tqdm(
                    f,
                    total=total_lines,
                    desc=desc,
                    unit="line",
                    dynamic_ncols=True,
                    leave=True,
                )
            for raw_line in line_iter:
                line = raw_line.strip()
                if not line or "#" not in line:
                    continue
                input_part, target_part = line.split("#", 1)
                input_values = self._parse_pipe_separated_ints(input_part)
                target_values = self._parse_pipe_separated_ints(target_part)
                input_ids = self._build_input_like_legacy(input_values)
                labels = self._build_labels_like_legacy(target_values)
                if not input_ids or not labels:
                    continue
                # input_ids/labels は dummy=0 を挟むだけなので、
                # local_max は元の値から直接計算できる（変換後リストを再走査しない）
                # input_ids: [dummy, (v+q), dummy, ...] -> max = max(input_values)+q
                # labels:    [(y+q), dummy, dummy, ...] -> max = y+q
                local_label_max = max(value for value in labels if value != self.dummy_token_id)
                local_max = max(max(input_values) + self.q, local_label_max)
                if local_max > self.max_token_id:
                    self.max_token_id = local_max
                sample = {"input_ids": input_ids, "labels": labels}
                self.samples.append(sample)
                if max_samples is not None and len(self.samples) >= max_samples:
                    break

    @staticmethod
    def _parse_pipe_separated_ints(text: str) -> List[int]:
        # split -> strip -> int のみ。append ループをやめて軽量化
        return [int(tok) for tok in (t.strip() for t in text.split("|")) if tok != ""]

    def _build_input_like_legacy(self, values: List[int]) -> List[int]:
        # [BOS, n1, '|', n2, '|', ..., nN, EOS] の位置関係をダミーで再現
        # モデル側 input_ids[:, 1::2] で実数値だけ取り出せる形にする。
        q = self.q
        dummy = self.dummy_token_id
        packed: List[int] = [dummy]
        for value in values:
            packed.append(value + q)
            packed.append(dummy)
        return packed

    def _build_labels_like_legacy(self, values: List[int]) -> List[int]:
        # 旧 labels=[y, '|', k, EOS] 相当の長さは保ちつつ、
        q = self.q
        dummy = self.dummy_token_id
        return [values[0] + q, dummy, dummy, dummy]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


class RawIdCollator:
    """可変長の input_ids / labels を最小限でパディングする collator。"""

    def __init__(self, input_pad_id: int = 0, label_pad_id: int = 0):
        self.input_pad_id = input_pad_id
        self.label_pad_id = label_pad_id

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch_size = len(batch)
        max_input_len = max(len(item["input_ids"]) for item in batch)
        max_label_len = max(len(item["labels"]) for item in batch)

        input_ids = torch.full(
            (batch_size, max_input_len), self.input_pad_id, dtype=torch.long
        )
        attention_mask = torch.zeros((batch_size, max_input_len), dtype=torch.long)
        labels = torch.full((batch_size, max_label_len), self.label_pad_id, dtype=torch.long)

        for i, item in enumerate(batch):
            current_input = torch.tensor(item["input_ids"], dtype=torch.long)
            current_labels = torch.tensor(item["labels"], dtype=torch.long)
            input_ids[i, : current_input.numel()] = current_input
            attention_mask[i, : current_input.numel()] = 1
            labels[i, : current_labels.numel()] = current_labels

        result: Dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        return result


def mod_diff(f_x: torch.Tensor, f_x_i: torch.Tensor, Q: int) -> torch.Tensor:
    """円環 Z_Q 上の距離（最短距離）を返す。"""
    if f_x.shape != f_x_i.shape:
        raise ValueError(f"Shape mismatch: {f_x.shape} vs {f_x_i.shape}")
    diff = torch.abs(f_x - f_x_i)
    return torch.minimum(diff, Q - diff)


def compute_metrics(eval_preds, ignore_index=-100):
        """Compute metrics at each prediction step.

        Args:
            eval_preds (tuple): (predictions, labels) where
                - predictions: shape (batch_size, seq_len)
                - labels: shape (batch_size, seq_len)
            ignore_index (int, optional): Label id to ignore. Defaults to -100.

        Returns:
            dict: Dictionary with accuracy metrics.
        """
        predictions, labels = eval_preds

        # Convert to tensors since inputs are often numpy arrays
        if isinstance(predictions, np.ndarray):
            predictions = torch.tensor(predictions)
        if isinstance(labels, np.ndarray):
            labels = torch.tensor(labels)

        # 先頭トークンのみで評価
        labels = labels[:, 0] 
        if predictions.ndim > 1:
            predictions = predictions[:, 0]

        embed_type = cfg["model"].get("embed_type", "token")

        if embed_type == "angular":
            # 円環距離ベースの閾値正答率:
            #   d(y_hat, y) = min(|y_hat-y|, q-|y_hat-y|)
            #   correct if d <= tau * Q
            q = int(cfg["train"]["q"])
            tau_list = [0.005, 0.01]
            label_mod = labels - q

            # ignore_index は元 labels で判定
            mask = labels != ignore_index
            if mask.sum().item() == 0:
                return {"angular_threshold_accuracy": 0.0}

            dist = mod_diff(predictions, label_mod, Q=q).to(torch.float32)
            acc_dict = {}
            for tau in tau_list:
                threshold = tau * float(q)
                correct = (dist <= threshold) & mask
                acc = correct.sum().item() / mask.sum().item()
                acc_dict[f"{tau}_threshold_accuracy"] = acc
            
            predictions = torch.round(predictions)
            predictions = predictions % q
            raund_precision = predictions.to(torch.long)
            correct = (raund_precision == label_mod) & mask
            acc = correct.sum().item() / mask.sum().item()
            acc_dict["success_rate"] = acc

            return acc_dict

        elif embed_type == "token":
            # token: 完全一致 + 許容誤差つき正答率
            mask = labels != ignore_index
            valid_count = mask.sum().item()
            if valid_count == 0:
                return {"success_rate": 0.0}

            # 連続値出力の可能性に備えて丸めてから評価
            predictions_rounded = torch.round(predictions).to(torch.long)
            label_values = labels.to(torch.long)

            # 従来の完全一致
            exact_correct = (predictions_rounded == label_values) & mask
            exact_acc = exact_correct.sum().item() / valid_count

            # 許容誤差つき評価（閾値は q 比率で指定）
            q = int(cfg["train"]["q"])
            tau_list = cfg["train"].get("token_tolerance_ratio_list", [0.1])
            abs_diff = torch.abs(predictions_rounded - label_values).to(torch.float32)

            acc_dict = {"success_rate": exact_acc}
            for tau in tau_list:
                threshold = float(tau) * float(q)
                correct = (abs_diff <= threshold) & mask
                acc_dict[f"{tau}_threshold_accuracy"] = correct.sum().item() / valid_count

            return acc_dict
        else:
            raise ValueError(f"Unsupported embed type: {embed_type}")

cfg = OmegaConf.load("config/config_train.yaml")

# wandbの初期化
wandb.init(
    project=cfg["wandb"]["project_name"],
    name=cfg["wandb"]["run_name"],
    entity=cfg["wandb"].get("entity", None),
    tags=cfg["wandb"]["tags"],
    config=OmegaConf.to_container(cfg, resolve=True)
)

# Apply sweep overrides to cfg so Trainer picks up wandb.config values
cfg = _merge_wandb_overrides(cfg)
_apply_sweep_dynamic_fields(cfg)

q = int(cfg["train"]["q"])
dummy_token_id = 0
print("start loading train dataset")
train_dataset = RawIdDataset(
    data_path=cfg["data"]["train_dataset_path"],
    q=q,
    dummy_token_id=dummy_token_id,
    max_samples=cfg["data"].get("num_train_samples", -1),
    show_progress=True,
    progress_desc="Loading train RawIdDataset",
)
print("end loading train dataset")
print("start loading test dataset")
test_dataset = RawIdDataset(
    data_path=cfg["data"]["test_dataset_path"],
    q=q,
    dummy_token_id=dummy_token_id,
    max_samples=cfg["data"].get("num_test_samples", -1),
    show_progress=True,
    progress_desc="Loading test RawIdDataset",
)
data_collator = RawIdCollator(input_pad_id=dummy_token_id, label_pad_id=dummy_token_id)
max_seen_token_id = max(train_dataset.max_token_id, test_dataset.max_token_id)
fallback_vocab_size = int(cfg["train"]["q"]) * (int(cfg["train"]["K"]) + 1) + 1
vocab_size = max(max_seen_token_id + 1, fallback_vocab_size)

# エンコーダー専用モデル構成
common_cfg = dict(
    d_model=cfg["model"]["hidden_size"],       # 隠れ層の次元数
    nhead=cfg["model"]["num_attention_heads"], # アテンションヘッド数
    num_encoder_layers=cfg["model"]["num_encoder_layers"],  # エンコーダ層の数
    num_decoder_layers=cfg["model"]["num_decoder_layers"],  # デコーダ層の数
    dim_feedforward=cfg["model"]["dim_feedforward"],  # フィードフォワード層の次元（一般的に4倍）
    dropout=cfg["model"]["dropout"],  # ドロップアウト率
    activation=cfg["model"]["activation"],  # 活性化関数
    layer_norm_eps=cfg["model"]["layer_norm_eps"],  # LayerNormのeps
    batch_first=True,  # バッチ次元を最初に
    norm_first=cfg["model"]["norm_first"],  # LayerNormを最初に適用するか
    bias=cfg["model"]["bias"],  # バイアスを使用
    vocab_size=vocab_size,  # 数値ID直接入力時の語彙サイズ
    # モデルが要求する固定入力長（旧: train.max_length）
    max_input_len=int(cfg["train"]["N"]),
    pad_token_id=0,
    eos_token_id=0,
    bos_token_id=0,
    use_positional_embedding=cfg["model"]["positional_embedding"],  # 位置エンコーディングの有無
    init_std=cfg["train"]["init_std"],  # 重みの初期化標準偏差
    linear_init_type=cfg["model"].get("linear_init_type", "normal"),
    embedding_init_type=cfg["model"].get("embedding_init_type", "normal"),
    weight_init=cfg["model"]["weight_init"],
)

encoder_only_cfg = dict(
    embed_type=cfg["model"]["embed_type"],
    q=cfg["train"]["q"],
    K=cfg["train"]["K"],
    loss_target_mode=cfg["train"]["mode"],
    choice_shifted_ratio=cfg["train"]["ratio"],
)

model_cfg = EncoderOnlyTransformerConfig(**common_cfg, **encoder_only_cfg)
model = TransformerForEncoderOnly(config=model_cfg)

# # 初期埋め込みを保存して後段の可視化で利用できるようにする
# init_emb_path = Path(cfg["train"]["save_dir"]) / "embedding_init.pt"
# init_emb_path.parent.mkdir(parents=True, exist_ok=True)
# try:
#     saved_init_path = model.save_initial_embeddings(init_emb_path)
#     wandb.config.update({"embedding_init_path": str(saved_init_path)}, allow_val_change=True)
# except Exception as exc:
#     print(f"[warn] failed to save initial embeddings: {exc}", file=sys.stderr)

args = TrainingArguments(
    output_dir=cfg["train"]["save_dir"],
    num_train_epochs=cfg["train"]["num_train_epochs"],
    logging_steps=cfg["train"]["logging_steps"],
    save_strategy=cfg["train"]["save_strategy"],
    save_steps=cfg["train"]["save_steps"],
    eval_strategy="steps",
    eval_steps=cfg["train"]["eval_steps"],
    per_device_train_batch_size=scaled_per_device_batch_size(cfg["train"]["per_device_train_batch_size"]),
    per_device_eval_batch_size=scaled_per_device_batch_size(cfg["train"]["per_device_eval_batch_size"]),
    seed=cfg["train"]["seed"],
    remove_unused_columns=False,
    optim=cfg["train"]["optimizer"],
    label_names=["labels"],
    report_to="wandb",  # wandbにレポート
    logging_dir=f"{cfg['train']['save_dir']}/logs",
    run_name=cfg["wandb"]["run_name"],
    learning_rate=cfg["train"]["learning_rate"],
    # 学習率スケジューラーの設定
    lr_scheduler_type=cfg["train"]["lr_scheduler_type"],  # コサイン降下スケジューラー
    warmup_ratio=cfg["train"]["warmup_ratio"],  # 線形ウォームアップ
    # 重み減衰の設定
    weight_decay=cfg["train"]["weight_decay"],  # L2正則化の強さ
    average_tokens_across_devices=False,
)

trainer = TrainerNoAutoEvalSave(
    args=args,
    model=model,
    processing_class=None,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

# train
results = trainer.train()
trainer.save_model()
metrics = results.metrics

# eval
eval_metrics = trainer.evaluate()
metrics.update(eval_metrics)
# `compute_metrics` の出力は evaluate では `eval_` 接頭辞付きになるため、
# 既存の downstream ログ互換のために `success_rate` にも詰める。
if "eval_success_rate" in eval_metrics:
    metrics["success_rate"] = eval_metrics["eval_success_rate"]
elif "success_rate" in eval_metrics:
    metrics["success_rate"] = eval_metrics["success_rate"]
else:
    metrics["success_rate"] = metrics.get("success_rate", 0.0)
# save metrics
trainer.save_metrics("all", metrics)

# wandbに最終メトリクスを記録
wandb.log({
    "final_success_rate": metrics["success_rate"],
    "final_test_loss": metrics.get("eval_loss", 0),
    "final_train_loss": metrics.get("train_loss", 0)
})

print(f'success rate on test set: {100*metrics["success_rate"]:.1f} %')

# wandbの終了
wandb.finish()