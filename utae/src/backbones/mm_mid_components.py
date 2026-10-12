"""独立 mid 路线使用的空间编码及显式掩码聚合。"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from src.backbones.mm_mid_memory import (
    chunked_frame_forward,
    chunked_phenology_logits,
    chunked_weighted_sum,
)
from src.backbones.utae import ConvBlock, DownConvBlock, PhenologyAttentionV2


def resize_valid(valid, size):
    """下采样保留单元内任一有效像素，上采样用最近邻。"""
    b, t, _, h, w = valid.shape
    if (h, w) == tuple(size):
        return valid
    if t == 0:
        return valid.new_zeros(b, 0, 1, *size)
    flat = valid.reshape(b * t, 1, h, w).float()
    if size[0] <= h and size[1] <= w:
        resized = F.adaptive_max_pool2d(flat, size)
    else:
        resized = F.interpolate(flat, size=size, mode="nearest")
    return resized.bool().reshape(b, t, 1, *size)


def framewise_forward(block, sequence, pad_mask, channels, size, chunk_size=0):
    """仅编码有效帧；空帧由显式掩码识别，不依赖特征是否全零。"""
    b, t, c, h, w = sequence.shape
    flat = sequence.reshape(b * t, c, h, w)
    selected = (~pad_mask).reshape(-1).nonzero(as_tuple=True)[0]
    if selected.numel() == 0:
        return sequence.new_zeros(b, t, channels, *size)
    if chunk_size > 0:
        output = chunked_frame_forward(block, flat, selected, channels, size, chunk_size)
        return output.reshape(b, t, channels, *size)
    encoded = block(flat.index_select(0, selected))
    output = encoded.new_zeros(b * t, channels, *size)
    output = output.index_copy(0, selected, encoded)
    return output.reshape(b, t, channels, *size)


def masked_phenology(aggregator, sequence, valid, time_chunk_size=0):
    """复用 P-MTSA 参数，同时排除逐像素缺测与全空时间序列。"""
    b, t, c, h, w = sequence.shape
    if t == 0:
        return sequence.new_zeros(b, c, h, w)
    safe = torch.where(valid, sequence, 0)
    if time_chunk_size > 0:
        temporal_logits = chunked_phenology_logits(aggregator, safe, time_chunk_size)
    else:
        temporal_logits = aggregator.attn_net(
            aggregator.conv3d(safe.permute(0, 2, 1, 3, 4))
        ).permute(0, 2, 1, 3, 4)
    logits = temporal_logits.masked_fill(~valid, torch.finfo(temporal_logits.dtype).min)
    weights = logits.softmax(dim=1) * valid.to(dtype=safe.dtype)
    weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
    accumulation_dtype = (
        torch.float32 if safe.dtype in {torch.float16, torch.bfloat16} else safe.dtype
    )
    weighted = (
        chunked_weighted_sum(safe, weights, time_chunk_size, accumulation_dtype)
        if time_chunk_size > 0
        else (safe * weights).sum(dim=1, dtype=accumulation_dtype)
    )
    mean = safe.sum(dim=1, dtype=accumulation_dtype) / valid.sum(dim=1).clamp_min(1)
    alpha = torch.sigmoid(aggregator.alpha)
    return (alpha * weighted + (1 - alpha) * mean).to(safe.dtype)


def masked_concat(fusion, features, validities):
    """缺失模态输入严格置零，全部缺失的位置消除投影偏置。"""
    safe = [torch.where(v, x, 0) for x, v in zip(features, validities)]
    available = torch.stack(validities).any(dim=0)
    return torch.where(available, fusion(*safe), 0)


class ShallowSpatialBranch(nn.Module):
    """只保留 in_conv、Down1、Down2 和两个浅层 P-MTSA。"""

    def __init__(
        self,
        input_dim,
        widths,
        sizes,
        k=4,
        s=2,
        p=1,
        norm="group",
        padding_mode="reflect",
        low_memory=False,
    ):
        super().__init__()
        if low_memory and norm != "group":
            raise ValueError("mid low-memory mode requires encoder_norm='group'")
        self.low_memory = bool(low_memory)
        self.widths, self.sizes = tuple(widths), tuple(sizes)
        self.in_conv = ConvBlock(
            [input_dim, widths[0], widths[0]],
            norm=norm,
            pad_value=None,
            padding_mode=padding_mode,
        )
        self.down_blocks = nn.ModuleList(
            DownConvBlock(
                widths[i],
                widths[i + 1],
                k,
                s,
                p,
                norm=norm,
                pad_value=None,
                padding_mode=padding_mode,
            )
            for i in range(2)
        )
        self.temporal_aggregator = nn.ModuleList(
            PhenologyAttentionV2(widths[i], stage_level=i) for i in range(2)
        )

    def forward(self, sequence, valid):
        if self.low_memory and self.training and torch.is_grad_enabled():
            return checkpoint(self._forward, sequence, valid, use_reentrant=False)
        return self._forward(sequence, valid)

    def _forward(self, sequence, valid):
        chunk = 8 if self.low_memory else 0
        safe = torch.where(valid, sequence, 0)
        pad_mask = ~valid.flatten(2).any(dim=-1)
        out = framewise_forward(self.in_conv, safe, pad_mask, self.widths[0], self.sizes[0], chunk)
        maps, masks = [out], [valid]
        for i, block in enumerate(self.down_blocks):
            stage_valid = resize_valid(valid, self.sizes[i + 1])
            stage_pad = ~stage_valid.flatten(2).any(dim=-1)
            out = framewise_forward(
                block, maps[-1], stage_pad, self.widths[i + 1], self.sizes[i + 1], chunk
            )
            out = torch.where(stage_valid, out, 0)
            maps.append(out)
            masks.append(stage_valid)
        skips = [
            masked_phenology(aggregator, maps[i], masks[i], time_chunk_size=chunk)
            for i, aggregator in enumerate(self.temporal_aggregator)
        ]
        skip_valid = [mask.any(dim=1) for mask in masks[:2]]
        return maps[-1], masks[-1], skips, skip_valid, pad_mask
