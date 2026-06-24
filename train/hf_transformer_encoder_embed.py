"""
hf_transformer.py から切り出したエンコーダー専用の設定クラスとモデル実装。
"""

import torch
import torch.nn as nn
from pathlib import Path
from typing import Optional, Dict, Any, Union, List

from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.utils import logging

from hf_transformer import TransformerConfig
from util.embed_utils import embed_circle, embed_inverse_add, embed_inverse_mul, embed_square, embed_fold_half_noflip, embed_fold_half_random, embed_fold_half_allrandom, embed_rotate_quarter_noflip, embed_rotate_quarter_allrandom, embed_rotate_quarter_origami_flip, embed_fold_quarter_noflip, embed_fold_quarter_allrandom, embed_fold_quarter_origami_flip
from transformers import PreTrainedModel


logger = logging.get_logger(__name__)
class CustomLoss(torch.nn.Module):
    def __init__(self, alpha, reduction="mean"):
        super(CustomLoss, self).__init__()
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, inputs, targets):
        pred_norm = torch.norm(inputs, dim=1, keepdim=True)
        loss_regularizer = (pred_norm**2) + 1.0 / (pred_norm**2 + 1e-6)
        loss_mse = torch.nn.functional.mse_loss(inputs.float(), targets.float(), reduction="none").sum(dim=1).unsqueeze(1)
        if self.reduction == "mean":
            loss_regularizer = torch.mean(loss_regularizer)
            loss_mse = torch.mean(loss_mse)
        return loss_mse + self.alpha * loss_regularizer

class MSELossMasked(nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets, mask):
        # maskは inputs と同じ形状 [batch, ..., 4] の 0 または 1 のテンソル
        # 1 の部分だけ勾配が計算・更新される
        loss_mse = torch.nn.functional.mse_loss(
            inputs.float(), targets.float(), reduction="none"
        )
        # マスクを掛けて不要な箇所の損失を 0 にする
        loss_mse = loss_mse * mask.float()
        loss_mse = loss_mse.sum(dim=-1).unsqueeze(-1)
        
        if self.reduction == "mean":
            # 単純なmeanだと 0 にした要素も母数に含まれてしまうため、
            # 有効な（maskが1の）要素数で割るのがより正確です
            loss_mse = loss_mse.sum() / mask.sum()
            
        return loss_mse

class MSELoss(torch.nn.Module):
    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss_mse = torch.nn.functional.mse_loss(
            inputs.float(), targets.float(), reduction="none"
        ).sum(dim=1).unsqueeze(1)
        if self.reduction == "mean":
            loss_mse = torch.mean(loss_mse)
        return loss_mse


class EncoderOnlyTransformerConfig(TransformerConfig):
    """
    エンコーダー専用モデル向け設定クラス。
    TransformerConfig を継承し、デフォルトのデコーダ層数を 0 に変更。
    """

    model_type = "transformer_encoder"

    def __init__(
        self,
        num_decoder_layers: int = 0,
        embed_type: str = "token",
        q: int = 257,
        K: int = 1,
        loss_target_mode: str = "normal",
        choice_n: int = 1,
        choice_shifted_ratio: float = 0.0,
        **kwargs,
    ):
        super().__init__(num_decoder_layers=num_decoder_layers, **kwargs)
        self.embed_type = embed_type
        self.q = q
        self.K = K
        self.loss_target_mode = loss_target_mode
        self.choice_n = choice_n
        self.choice_shifted_ratio = choice_shifted_ratio


class TransformerForEncoderOnly(PreTrainedModel):
    """エンコーダーのみの Transformer 実装。"""

    config_class = EncoderOnlyTransformerConfig

    def __init__(self, config: EncoderOnlyTransformerConfig):
        super().__init__(config)
        self.seed = config.seed
        torch.manual_seed(self.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)

        self.config = config
        self.q = config.q
        self.K = config.K
        self.r = config.choice_shifted_ratio
        self.choice_n = config.choice_n

        # 埋め込み層
        if self.config.embed_type == "token":
            self.embedding = nn.Embedding(self.q, config.d_model)
        elif self.config.embed_type == "angular":
            self.embedding = nn.Linear(2, config.d_model)
        else:
            raise ValueError(f"Unsupported embed type: {self.config.embed_type}")

        # self.embedding_table = embed_circle(self.q, config.d_model, config.init_std)
        # self.embedding_table = self.embedding_table.to("cuda")
        # 位置エンべディング（設定に応じて有無を切り替え）
        self.positional_embedding = None


        # エンコーダー層: PyTorch 実装のみを使用し、必要に応じて注意機構だけ差し替える
        encoder_layer_kwargs = dict(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation=config.activation,
            layer_norm_eps=config.layer_norm_eps,
            batch_first=config.batch_first,
            norm_first=config.norm_first,
            bias=config.bias,
        )

        encoder_layer = nn.TransformerEncoderLayer(**encoder_layer_kwargs)

        encoder_norm = nn.LayerNorm(
            config.d_model,
            eps=config.layer_norm_eps,
            bias=self.config.bias,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers,
            norm=encoder_norm,
        )

        # 固定長の出力層を作成
        if self.config.embed_type == "token":
            self.lm_head = nn.Linear(
                config.d_model, self.K*self.q, bias=self.config.bias
            )
        elif self.config.embed_type == "angular":
            if self.config.loss_target_mode == "mod_Kq":
                self.lm_head = nn.Linear(
                    config.d_model, 4, bias=self.config.bias
                )
            else:
                self.lm_head = nn.Linear(
                    config.d_model, 2, bias=self.config.bias
            )
        else:
            raise ValueError(f"Unsupported embed type: {self.config.embed_type}")
        
        # 重みの初期化
        if self.config.weight_init:
            self.apply(self._init_weights)

        self.tie_weights()

        self.special_token_exists = False
    
    def _init_weights(self, module):
        """重みの初期化メソッド"""
        if isinstance(module, nn.Linear):
            initialized = self._init_linear_weights(module)
            if initialized and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            self._init_embedding_weights(module)
        elif isinstance(module, nn.LayerNorm):
            # LayerNormの重みを1、バイアスを0で初期化
            if module.bias is not None: 
                module.bias.data.zero_()
            if module.weight is not None:
                module.weight.data.fill_(1.0)

    def _init_linear_weights(self, module: nn.Linear) -> None:
        init_type = getattr(self.config, "linear_init_type", "normal")
        if init_type == "none":
            return False
        if init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        else:
            raise ValueError(f"Unsupported linear_init_type: {init_type}")
        return True

    def _init_embedding_weights(self, module: nn.Embedding) -> None:
        init_type = getattr(self.config, "embedding_init_type", "normal")
        if init_type == "none":
            return
        elif init_type == "normal":
            nn.init.normal_(module.weight, mean=0.0, std=self.config.init_std)
        elif init_type == "xavier_uniform":
            nn.init.xavier_uniform_(module.weight)
        elif init_type == "xavier_normal":
            nn.init.xavier_normal_(module.weight)
        else:
            # embed_utils に委譲して初期化
            embed_func_map = {
                "circle": embed_circle,
                "inverse_add": embed_inverse_add,
                "inverse_mul": embed_inverse_mul,
                "square": embed_square,
                "fold_half_noflip": embed_fold_half_noflip,
                "fold_half_random": embed_fold_half_random,
                "fold_half_allrandom": embed_fold_half_allrandom,
                "rotate_quarter_noflip": embed_rotate_quarter_noflip,
                "rotate_quarter_allrandom": embed_rotate_quarter_allrandom,
                "rotate_quarter_origami_flip": embed_rotate_quarter_origami_flip,
                "fold_quarter_noflip": embed_fold_quarter_noflip,
                "fold_quarter_allrandom": embed_fold_quarter_allrandom,
                "fold_quarter_origami_flip": embed_fold_quarter_origami_flip,
            }
            func = embed_func_map.get(init_type)
            if func is None:
                raise ValueError(f"Unsupported embedding_init_type: {init_type}")

            q = self.config.q
            d_model = self.config.d_model
            weight = func(q, d_model) * self.config.init_std
            with torch.no_grad():
                # 通常はmodule.weight[q:2*q]へ書き込みだが、埋め込みサイズによる
                # module.weightが2q以上のサイズである前提
                module.weight[q:2*q].copy_(weight)

    def save_initial_embeddings(self, save_path: Union[str, Path]) -> Path:
        """
        現在の埋め込み重みをファイルに保存する。
        可視化などの再利用を想定して、メタデータも併せて保存する。
        """
        path = Path(save_path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            torch.save(
                {
                    "embedding_weight": self.embedding.weight.detach().cpu().clone(),
                    "embed_type": self.config.embed_type,
                    "q": getattr(self.config, "q", None),
                    "d_model": getattr(self.config, "d_model", None),
                    "init_std": getattr(self.config, "init_std", None),
                    "embedding_init_type": getattr(self.config, "embedding_init_type", None),
                    "seed": getattr(self.config, "seed", None),
                },
                path,
            )
            logger.info(f"Saved initial embedding weights to {path}")
        except Exception as exc:
            logger.warning(f"Failed to save initial embedding weights to {path}: {exc}")

        return path

    def _prepare_key_padding_mask(self, mask: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if mask is None:
            return None
        if mask.dtype == torch.bool:
            key_padding_mask = ~mask
        else:
            key_padding_mask = mask == 0
        return key_padding_mask.to(dtype=torch.bool, device=mask.device)
    
    def _generate_causal_mask(self, size: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(size, size, device=device, dtype=torch.bool), diagonal=1)

    def input_to_cos_sin(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        入力トークン ID（実値が q 加算されている前提）を角度に写像し、(sin, cos) の 2 次元表現に変換する。
        """
        if input_ids is None:
            raise ValueError("input_idsが必要です")

        ids = input_ids.to(dtype=torch.float32)
        ids = ids - self.q
        modulus = self.q

        angles = ids * 2 * torch.pi / modulus

        sin_component = torch.sin(angles)
        cos_component = torch.cos(angles)

        return torch.stack((cos_component, sin_component), dim=-1)
    
    def input_to_cos_sin_Kq(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """
        入力トークン ID（実値が q 加算されている前提）を角度に写像し、(sin, cos) の 2 次元表現に変換する。
        """
        if input_ids is None:
            raise ValueError("input_idsが必要です")

        ids = input_ids.to(dtype=torch.float32)
        ids = ids - self.q

        Kq = torch.as_tensor(self.K * self.q, dtype=ids.dtype, device=ids.device)

        modulus = Kq

        angles = ids * 2 * torch.pi / modulus

        sin_component = torch.sin(angles)
        cos_component = torch.cos(angles)

        return torch.stack((cos_component, sin_component), dim=-1)

    def cos_sin_to_input(self, logits: torch.Tensor) -> torch.LongTensor:
        """
        (cos, sin) ロジットから入力トークン ID（q 加算済み）を復元する。
        """
        if logits is None:
            raise ValueError("logitsが必要です")
        if logits.size(-1) != 2:
            raise ValueError("logitsの最後の次元は2である必要があります")

        cos_component = logits[..., 0]
        sin_component = logits[..., 1]

        angles = torch.atan2(sin_component, cos_component) + 2 * torch.pi % (2 * torch.pi)

        values = (torch.round(angles * self.q / (2 * torch.pi))) % self.q + self.q

        return values.to(torch.long)
    
    def cos_sin_to_input_float(self, logits: torch.Tensor) -> torch.Tensor:
        """
        (cos, sin) ロジットから入力トークン ID相当の連続値（float）を復元する。
        評価時に丸め誤差を含む連続値で距離評価したいケース向け。
        """
        if logits is None:
            raise ValueError("logitsが必要です")
        if logits.size(-1) != 2:
            raise ValueError("logitsの最後の次元は2である必要があります")

        cos_component = logits[..., 0]
        sin_component = logits[..., 1]

        angles = torch.atan2(sin_component, cos_component)
        angles = (angles + 2 * torch.pi) % (2 * torch.pi)

        values = angles * self.q / (2 * torch.pi)

        return values

    def cos_sin_to_input_Kq(self, logits: torch.Tensor) -> torch.LongTensor:
        """
        (cos, sin) ロジットから入力トークン ID（q 加算済み）を復元する。
        """
        if logits is None:
            raise ValueError("logitsが必要です")
        if logits.size(-1) != 2:
            raise ValueError("logitsの最後の次元は2である必要があります")

        cos_component = logits[..., 0]
        sin_component = logits[..., 1]

        # 1. 角度を取得し、[-π, π] の範囲を [0, 2π) に正規化する
        angles = torch.atan2(sin_component, cos_component)
        angles = (angles + 2 * torch.pi) % (2 * torch.pi)

        # 2. 角度から元の ids (入力トークン ID - q) を復元する
        # 順伝播の逆算: ids = angle * modulus / 2π
        Kq = self.K * self.q
        ids_restored = torch.round(angles * Kq / (2 * torch.pi))

        # 3. 丸めによる境界値 (Kq) を 0 に戻し、q を加算して入力トークンIDに戻す
        ids_restored = ids_restored % self.q
        values = ids_restored + self.q

        # 4. LongTensor(int64) 型にキャストして返す
        return values.to(torch.long)

    def cos_sin_to_input_Kq_float(self, logits: torch.Tensor) -> torch.Tensor:
        """
        (cos, sin) ロジットから入力トークンID相当の連続値（float）を復元する。
        評価時に丸め誤差を含む連続値で距離評価したいケース向け。
        """
        if logits is None:
            raise ValueError("logitsが必要です")
        if logits.size(-1) != 2:
            raise ValueError("logitsの最後の次元は2である必要があります")

        cos_component = logits[..., 0]
        sin_component = logits[..., 1]
        angles = torch.atan2(sin_component, cos_component)
        angles = (angles + 2 * torch.pi) % (2 * torch.pi)

        Kq = torch.as_tensor(self.K * self.q, dtype=angles.dtype, device=angles.device)
        ids_restored = angles * Kq / (2 * torch.pi)
    
        return ids_restored

    def _auto_choice_weights(self, n: int, device, **kwargs):
        if n <= 0:
            return None
        r = float(getattr(self.config, "choice_shifted_ratio", 0.0))
        r = max(0.0, min(1.0, r))
        w0 = 1.0 - r
        wi = r / float(n)
        w = torch.tensor([w0] + [wi] * n, device=device, dtype=torch.float32)
        if float(w.sum()) <= 0.0:
            w = torch.ones_like(w)
        return w

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> Dict[str, Any]:

        return_dict = (
            return_dict
            if return_dict is not None
            else getattr(self.config, "return_dict", getattr(self.config, "use_return_dict", True))
        )
        output_hidden_states = (
            output_hidden_states
            if output_hidden_states is not None
            else getattr(self.config, "output_hidden_states", False)
        )

        if input_ids is None:
            raise ValueError("input_idsが必要です")
        
        if not self.special_token_exists:
            # 一時的な処置: 入力の偶数番目(1始まり)を取り除き、通常トークンのみを使用
            input_ids = input_ids[:, 1::2]

            if attention_mask is not None:
                attention_mask = attention_mask[:, 1::2]

        batch_size, seq_len = input_ids.shape

        # 埋め込みの計算
        if self.config.embed_type == "token":
            input_ids = input_ids - self.q
            embeddings = self.embedding(input_ids)

        elif self.config.embed_type == "angular":
            if self.config.loss_target_mode == "mod_Kq":
                input_ids_cos_sin = self.input_to_cos_sin_Kq(input_ids)
                embeddings = self.embedding(input_ids_cos_sin)
            else:
                input_ids_cos_sin = self.input_to_cos_sin(input_ids)
                embeddings = self.embedding(input_ids_cos_sin)

        else:
            raise ValueError(f"Unsupported embed type: {self.config.embed_type}")

        encoder_key_padding_mask = self._prepare_key_padding_mask(attention_mask)

        # エンコーダー層を順次適用
        x = embeddings
        x = self.encoder(x, src_key_padding_mask=encoder_key_padding_mask)

        # 出力ヘッド直前の表現
        pre_head_repr = x

        if seq_len != self.config.max_input_len:
            raise ValueError(
                f"入力の長さが正しくありません。期待される長さ: {self.config.max_input_len}, 実際の長さ: {seq_len}"
            )
        
        pooled = torch.max(x, dim=1)[0]
        
        logits = self.lm_head(pooled)

        if not self.training and self.config.embed_type == "token":
            logits = logits[:, :self.q]
        
        if self.config.embed_type == "token":
            # 損失の計算
            loss = None
            if labels is not None:
                labels_no_eos = labels[:, 0] - self.q
                mode = getattr(self.config, "loss_target_mode", "normal")
                if mode not in {"normal", "sparse", "mod_Kq"}:
                    raise ValueError(
                        f"Unsupported loss_target_mode for token embed: {mode}"
                    )

                if mode == "mod_Kq" and self.training:
                    labels_K = labels[:, 2] - self.q

                    n = getattr(self.config, "choice_n", 1)
                    B = labels_no_eos.size(0)
                    i = torch.arange(
                        max(self.K - (n - 1), 2),
                        self.K + 1,
                        device=labels_no_eos.device,
                    )
                    choice_list = i.unsqueeze(0).expand(B, -1) * self.q
                    choice_list = torch.cat(
                        [
                            torch.ones(
                                B,
                                1,
                                device=choice_list.device,
                                dtype=choice_list.dtype,
                            )
                            * self.q,
                            choice_list,
                        ],
                        dim=1,
                    )  # [B, n+1]

                    w = self._auto_choice_weights(
                        n=n, device=labels_no_eos.device, **kwargs
                    )  # [n+1]
                    rand_choice = torch.multinomial(w.expand(B, -1), 1).squeeze(1)  # [B]

                    picked_shift = choice_list.gather(
                        1, rand_choice.unsqueeze(1)
                    ).squeeze(1)  # [B]

                    labels_add = labels_no_eos + labels_K * self.q
                    labels_no_eos = labels_add % picked_shift


                loss_fct = nn.CrossEntropyLoss()
                n_classes = logits.size(-1)
                if torch.any((labels_no_eos < 0) | (labels_no_eos >= n_classes)):
                    bad_min = int(labels_no_eos.min().item())
                    bad_max = int(labels_no_eos.max().item())
                    raise ValueError(
                        f"labels_no_eos out of range for CE: min={bad_min}, max={bad_max}, n_classes={n_classes}, mode={mode}"
                    )
                loss = loss_fct(logits.view(-1, logits.size(-1)), labels_no_eos.view(-1))

            logits_argmax = torch.argmax(logits, dim=-1).squeeze(-1)
            logits_argmax = logits_argmax + self.q

        elif self.config.embed_type == "angular":
            mode = getattr(self.config, "loss_target_mode", "normal")
            if mode not in {"normal", "sparse", "mod_Kq"}:
                raise ValueError(
                    f"Unsupported loss_target_mode for angular embed: {mode}"
                )
            loss = None
            mask_one = torch.ones_like(logits[:, :2])
            mask_zero = torch.zeros_like(logits[:, :2])
            mask_q = torch.cat([mask_one, mask_zero], dim=-1)
            mask_Kq = torch.cat([mask_zero, mask_one], dim=-1)
            mask = mask_q

            if labels is not None:
                labels_no_eos = labels[:, 0] - self.q
                if mode == "mod_Kq":
                    labels_K = labels[:, 2] - self.q

                    if self.training:
                        n = 1
                        B = labels_no_eos.size(0)
                        i = torch.arange(
                            max(self.K - (n - 1), 2),
                            self.K + 1,
                            device=labels_no_eos.device,
                        )
                        choice_list = i.unsqueeze(0).expand(B, -1) * self.q
                        choice_list = torch.cat(
                            [
                                torch.ones(
                                    B,
                                    1,
                                    device=choice_list.device,
                                    dtype=choice_list.dtype,
                                )
                                * self.q,
                                choice_list,
                            ],
                            dim=1,
                        )  # [B, n+1]

                        w = self._auto_choice_weights(
                            n=n, device=labels_no_eos.device, **kwargs
                        )  # [n+1]
                        rand_choice = torch.multinomial(w.expand(B, -1), 1).squeeze(1)

                        picked_shift = choice_list.gather(
                            1, rand_choice.unsqueeze(1)
                        ).squeeze(1)

                        labels_add = labels_no_eos + labels_K * self.q
                        labels_no_eos = labels_add % picked_shift

                        mask = torch.zeros_like(logits)
                        idx_q = rand_choice == 0
                        idx_Kq = rand_choice == 1
                        mask[idx_q, ..., :2] = 1
                        mask[idx_Kq, ..., 2:] = 1

                if mode == "mod_Kq":
                    labels_q = self.input_to_cos_sin(labels_no_eos+self.q)
                    labels_Kq = self.input_to_cos_sin_Kq(labels_no_eos+self.q)
                    labels_no_eos = torch.cat([labels_q, labels_Kq], dim=-1)
                else:
                    labels_no_eos = self.input_to_cos_sin(labels_no_eos+self.q)
                  
                if mode == "mod_Kq":
                    loss_fct = MSELossMasked(reduction="none")
                    elementwise_loss = loss_fct(logits.float(), labels_no_eos.float(), mask=mask)
                    loss = elementwise_loss.mean()
                elif mode == "sparse":
                    loss_fct = CustomLoss(alpha=0.01, reduction="none")
                    elementwise_loss = loss_fct(logits.float(), labels_no_eos.float())
                    loss = elementwise_loss.mean()
                elif mode == "normal":
                    loss_fct = MSELossMasked(reduction="none")
                    elementwise_loss = loss_fct(logits.float(), labels_no_eos.float(), mask=mask_one)
                    loss = elementwise_loss.mean()

            # evaluate() の predictions では連続値を返し、閾値距離で評価しやすくする
            logits_argmax = self.cos_sin_to_input_float(logits[:, :2])

        encoder_last_hidden_state = pre_head_repr if output_hidden_states else None

        return Seq2SeqLMOutput(
            loss=loss,
            logits=logits_argmax,
            past_key_values=None,
            decoder_hidden_states=None,
            decoder_attentions=None,
            cross_attentions=None,
            encoder_last_hidden_state=encoder_last_hidden_state,
            encoder_hidden_states=None,
            encoder_attentions=None,
        )

    def generate(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: int = 100,
        num_beams: int = 1,
        temperature: float = 1.0,
        do_sample: bool = False,
        pad_token_id: Optional[int] = None,
        eos_token_id: Optional[int] = None,
        **kwargs,
    ) -> torch.LongTensor:
        """
        エンコーダー専用モデル用のテキスト生成メソッド。
        注意: この実装はエンコーダー専用モデルでは適切ではない可能性があります。
        """
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id

        if not self.special_token_exists:
            input_ids = input_ids[:, 1::2]
            if attention_mask is not None:
                attention_mask = attention_mask[:, 1::2]

        batch_size = input_ids.shape[0]
        device = input_ids.device

        # エンコーダー専用モデルでは、入力シーケンス全体を処理して単一の出力を生成
        seq_len = input_ids.shape[1]
        if self.config.embed_type == "token":
            embeddings = self.embedding(input_ids)
        elif self.config.embed_type == "angular":
            input_ids_cos_sin = self.input_to_cos_sin_Kq(input_ids)
            embeddings = self.embedding(input_ids_cos_sin)
        else:
            raise ValueError(f"Unsupported embed type: {self.config.embed_type}")

        encoder_key_padding_mask = self._prepare_key_padding_mask(attention_mask)

        # エンコーダー層を順次適用
        x = embeddings
        x = self.encoder(x, src_key_padding_mask=encoder_key_padding_mask)

        if seq_len != self.config.max_input_len:
            raise ValueError(
                f"入力の長さが正しくありません。期待される長さ: {self.config.max_input_len}, 実際の長さ: {seq_len}"
            )

        # 固定の線形層を適用
        pooled = torch.max(x, dim=1)[0]
        logits = self.lm_head(pooled)

        if self.config.embed_type == "token":
            logits = logits[:, :self.q]
            if do_sample:
                logits = logits / temperature
                next_token = torch.multinomial(torch.softmax(logits, dim=-1), 1) + self.q
            else:
                next_token = torch.argmax(logits, dim=-1, keepdim=True) + self.q
        elif self.config.embed_type == "angular":
            logits = logits[:, :2]
            if do_sample:
                raise NotImplementedError("Sampling is not supported for angular embeddings.")
            next_token = self.cos_sin_to_input_float(logits).unsqueeze(-1)
        else:
            raise ValueError(f"Unsupported embed type: {self.config.embed_type}")

        return next_token
