"""1016 完整异步主路：以逐像素掩码 L-TAE 替换共享 ConvLSTM。"""

from functools import partial

import torch
from torch import nn
from torch.nn import functional as F

from src.backbones.ltae import LTAE2d, ScaledDotProductAttention
from src.backbones.mm_ucpmamba_async import MMUCPMambaAsync


class _StrictMaskedAttention(ScaledDotProductAttention):
    """仅接收至少有一个有效日期的序列；无效日期权重严格为零。"""

    def forward(self, q, k, v, pad_mask=None, return_comp=False):
        scores = torch.matmul(q.unsqueeze(1), k.transpose(1, 2)) / self.temperature
        if pad_mask is not None:
            scores = scores.masked_fill(pad_mask.unsqueeze(1), -torch.inf)
        attention = self.dropout(self.softmax(scores))
        output = torch.matmul(attention, v)
        if return_comp:
            return output, attention, scores
        return output, attention


class WindowLTAE(LTAE2d):
    """复用 LTAE 参数模块，显式排除逐像素缺测和全空序列。

    输入归一化仅统计有效日期；全空像素不进入 MLP BatchNorm。
    仅一个有效像素时使用既有 BN running statistics，不更新其统计。
    """

    def __init__(self, input_dim, input_size, hidden_dim, kernel_size,
                 n_head=16, d_model=256, d_k=4):
        del input_size, kernel_size  # 兼容原共享时序单元的构造接口。
        values = (input_dim, hidden_dim, n_head, d_model, d_k)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values):
            raise ValueError("L-TAE dimensions must be positive integers")
        if any(width % n_head for width in (input_dim, hidden_dim, d_model)):
            raise ValueError("L-TAE input/output/model widths must be divisible by n_head")
        super().__init__(in_channels=input_dim, n_head=n_head, d_k=d_k,
                         mlp=[d_model, hidden_dim], d_model=d_model)
        original = self.attention_heads.attention
        self.attention_heads.attention = _StrictMaskedAttention(
            temperature=original.temperature, attn_dropout=original.dropout.p,
        )
        self.output_dim = hidden_dim

    def _normalize_valid(self, sequence, valid):
        """与原输入 GroupNorm 同一分组，只从有效 token 计算均值和方差。"""
        n, t, c = sequence.shape
        groups = self.in_norm.num_groups
        dtype = sequence.dtype
        stats = sequence.float() if dtype in (torch.float16, torch.bfloat16) else sequence
        grouped = stats.transpose(1, 2).reshape(n, groups, c // groups, t)
        mask = valid[:, None, None, :]
        count = valid.sum(1)[:, None, None, None] * (c // groups)
        mean = grouped.sum((2, 3), keepdim=True) / count
        centered = torch.where(mask, grouped - mean, 0)
        variance = centered.square().sum((2, 3), keepdim=True) / count
        normalized = centered * torch.rsqrt(variance + self.in_norm.eps)
        normalized = normalized.reshape(n, c, t).transpose(1, 2).to(dtype)
        if self.in_norm.affine:
            normalized = normalized * self.in_norm.weight + self.in_norm.bias
        return torch.where(valid[:, :, None], normalized, 0)

    def _apply_mlp(self, features):
        for layer in self.mlp:
            if isinstance(layer, nn.BatchNorm1d) and layer.training and features.shape[0] == 1:
                features = F.batch_norm(features, layer.running_mean, layer.running_var,
                                        layer.weight, layer.bias, training=False, eps=layer.eps)
            else:
                features = layer(features)
        return features

    def forward(self, x, batch_positions=None, pad_mask=None, valid_mask=None,
                return_att=False):
        if x.ndim != 5 or not x.is_floating_point():
            raise ValueError("L-TAE input must be floating [B,T,C,H,W]")
        b, t, c, h, w = x.shape
        if c != self.in_channels or min(b, t, h, w) <= 0:
            raise ValueError("L-TAE input dimensions are invalid")
        if batch_positions is None or batch_positions.shape != (b, t):
            raise ValueError("L-TAE requires window dates with shape [B,T]")
        if batch_positions.device != x.device or batch_positions.dtype == torch.bool \
                or batch_positions.is_complex():
            raise ValueError("L-TAE dates must be real numbers on the input device")
        if valid_mask is None:
            valid_mask = torch.ones(b, t, 1, h, w, dtype=torch.bool, device=x.device)
        if valid_mask.shape != (b, t, 1, h, w) or valid_mask.dtype != torch.bool \
                or valid_mask.device != x.device:
            raise ValueError("L-TAE valid_mask must be bool [B,T,1,H,W] on the input device")
        if pad_mask is not None:
            if pad_mask.shape != (b, t) or pad_mask.dtype != torch.bool \
                    or pad_mask.device != x.device:
                raise ValueError("L-TAE pad_mask must be bool [B,T] on the input device")
            valid_mask = valid_mask & ~pad_mask[:, :, None, None, None]
        if not torch.isfinite(batch_positions[valid_mask.flatten(2).any(-1)]).all():
            raise ValueError("L-TAE valid window dates must be finite")
        valid = valid_mask[:, :, 0].permute(0, 2, 3, 1).reshape(b * h * w, t)
        pixels = x.permute(0, 3, 4, 1, 2).reshape(b * h * w, t, c)
        pixels = torch.where(valid[:, :, None], pixels, 0)
        active = valid.any(1)
        output = x.new_zeros(b * h * w, self.output_dim) + pixels[0, 0, 0] * 0
        full_attention = x.new_zeros(self.n_head, b * h * w, t) if return_att else None
        if active.any():
            active_valid = valid[active]
            features = self._normalize_valid(pixels[active], active_valid)
            if self.inconv is not None:
                features = self.inconv(features.transpose(1, 2)).transpose(1, 2)
            positions = batch_positions[:, None, None, :].expand(b, h, w, t)
            positions = positions.reshape(b * h * w, t)[active].float()
            positions = torch.where(active_valid, positions, 0)
            if self.positional_encoder is not None:
                # 原日期编码器的 denom 非 buffer，显式同步设备以支持 model.to()。
                self.positional_encoder.denom = self.positional_encoder.denom.to(x.device)
                features = features + self.positional_encoder(positions).to(features.dtype)
            features, attention = self.attention_heads(features, pad_mask=~active_valid)
            features = features.permute(1, 0, 2).reshape(features.shape[1], -1)
            features = self.dropout(self._apply_mlp(features))
            features = self.out_norm(features)
            output[active] = features
            if return_att:
                full_attention[:, active] = attention
        output = output.reshape(b, h, w, self.output_dim).permute(0, 3, 1, 2)
        if return_att:
            attention = full_attention.reshape(self.n_head, b, h, w, t)
            return output, attention.permute(0, 1, 4, 2, 3)
        return output


class MMUPTAE(MMUCPMambaAsync):
    """完整 A+B 与七 P-MTSA 保留；L-TAE 输出 64 通道后走原投影。"""

    def __init__(self, *args, n_head=16, d_model=256, d_k=4, **kwargs):
        if "temporal_encoder_class" in kwargs:
            raise ValueError("mm-uptae requires WindowLTAE")
        temporal = partial(WindowLTAE, n_head=n_head, d_model=d_model, d_k=d_k)
        super().__init__(*args, temporal_encoder_class=temporal, **kwargs)

    def _encode_temporal(self, deep, pad_mask, valid_mask=None):
        dates = self.aligner.centers[None].expand(deep.shape[0], -1)
        aggregated = self.temporal_encoder(deep, batch_positions=dates,
                                           pad_mask=pad_mask, valid_mask=valid_mask)
        return self.out_convlstm(aggregated)
