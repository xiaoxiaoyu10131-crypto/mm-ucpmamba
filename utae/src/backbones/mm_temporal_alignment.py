"""按固定日期窗口对齐不同模态的特征，逐像素保留有效观测。"""

import math
from typing import Optional

import torch
from torch import nn


def normalize_valid_mask(
    sequence: torch.Tensor,
    pad_mask: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """返回 [B,T,1,H,W] 有效掩码；True 有效，padding 始终无效。"""
    if sequence.ndim != 5:
        raise ValueError("sequence must have shape [B,T,C,H,W]")
    b, t, _, h, w = sequence.shape
    if pad_mask.shape != (b, t) or pad_mask.dtype != torch.bool:
        raise ValueError("pad_mask must be bool with shape [B,T]")
    if pad_mask.device != sequence.device:
        raise ValueError("pad_mask and sequence must be on the same device")
    frame_valid = ~pad_mask[:, :, None, None, None]
    if valid_mask is None:
        return frame_valid.expand(b, t, 1, h, w)
    if valid_mask.dtype != torch.bool:
        raise ValueError("valid_mask must have bool dtype")
    if valid_mask.device != sequence.device:
        raise ValueError("valid_mask and sequence must be on the same device")
    if valid_mask.shape == (b, t):
        valid_mask = valid_mask[:, :, None, None, None]
    elif valid_mask.shape != (b, t, 1, h, w):
        raise ValueError("valid_mask must have shape [B,T] or [B,T,1,H,W]")
    return (valid_mask & frame_valid).expand(b, t, 1, h, w)


class DateWindowAligner(nn.Module):
    """在全数据共用的 [start_day,end_day) 内按固定窗口作 masked mean。

    日期坐标须由调用者统一；不根据一个 batch 的日期范围决定窗口。
    空窗口输出严格为零；counts 为每个像素的有效观测数（int64）。
    """

    edges: torch.Tensor
    centers: torch.Tensor

    def __init__(self, num_bins=24, start_day=None, end_day=None):
        super().__init__()
        if isinstance(num_bins, bool) or not isinstance(num_bins, int) or num_bins <= 0:
            raise ValueError("num_bins must be a positive integer")
        try:
            start_day, end_day = float(start_day), float(end_day)
        except (TypeError, ValueError) as exc:
            raise ValueError("finite start_day and end_day are required") from exc
        if not math.isfinite(start_day) or not math.isfinite(end_day) or start_day >= end_day:
            raise ValueError("start_day and end_day must be finite with start_day < end_day")
        self.num_bins = num_bins
        self.start_day = start_day
        self.end_day = end_day
        edges = torch.linspace(start_day, end_day, num_bins + 1, dtype=torch.float64)
        self.register_buffer("edges", edges, persistent=True)
        self.register_buffer("centers", (edges[:-1] + edges[1:]) / 2, persistent=True)

    def _apply(self, fn, recurse=True):
        # model.half()/float() 不得降低日期窗口精度；只跟随设备迁移。
        super()._apply(fn, recurse=recurse)
        self.edges = torch.linspace(
            self.start_day,
            self.end_day,
            self.num_bins + 1,
            dtype=torch.float64,
            device=self.edges.device,
        )
        self.centers = (self.edges[:-1] + self.edges[1:]) / 2
        return self

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        expected = torch.linspace(
            self.start_day,
            self.end_day,
            self.num_bins + 1,
            dtype=torch.float64,
        )
        for name, target in (
            ("edges", expected),
            ("centers", (expected[:-1] + expected[1:]) / 2),
        ):
            saved = state_dict.get(prefix + name)
            if saved is not None and (
                not isinstance(saved, torch.Tensor)
                or not saved.is_floating_point()
                or saved.shape != target.shape
                or not torch.equal(saved.detach().cpu().to(torch.float64), target)
            ):
                error_msgs.append(
                    f"{prefix}{name}: checkpoint date windows do not match "
                    f"configured num_bins={self.num_bins}, "
                    f"start_day={self.start_day}, end_day={self.end_day}"
                )
                return
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def validate_dates(self, positions, valid):
        """核验原始有效观测日期；返回无效帧日期已替换的 float64 张量。"""
        if (
            not isinstance(valid, torch.Tensor)
            or valid.ndim != 5
            or valid.shape[2] != 1
            or valid.dtype != torch.bool
        ):
            raise ValueError("valid must be bool with shape [B,T,1,H,W]")
        b, t, _, h, w = valid.shape
        if not isinstance(positions, torch.Tensor):
            raise ValueError("positions must be a tensor with shape [B,T]")
        if positions.shape != (b, t):
            raise ValueError("positions must have shape [B,T]")
        if positions.device != valid.device:
            raise ValueError("positions and valid must be on the same device")
        if not positions.is_floating_point() and positions.dtype not in (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
        ):
            raise ValueError("positions must contain real numeric dates")
        frame_valid = valid.reshape(b, t, h * w).any(dim=-1)
        dates = positions.to(dtype=torch.float64)
        finite = torch.isfinite(dates)
        if bool((frame_valid & ~finite).any()):
            raise ValueError("positions contain non-finite dates for valid observations")
        in_range = (dates >= self.start_day) & (dates < self.end_day)
        if bool((frame_valid & ~in_range).any()):
            raise ValueError("valid observation dates must be in [start_day,end_day)")
        # 无效帧的日期可能是 NaN；在 bucketize 前替换，避免不合法索引。
        return torch.where(frame_valid, dates, dates.new_full((), self.start_day))

    def forward(self, sequence, positions, pad_mask, valid_mask=None):
        """返回 pooled、valid、counts，形状为 [B,K,C/1,H,W]。"""
        valid = normalize_valid_mask(sequence, pad_mask, valid_mask)
        b, t, c, h, w = sequence.shape
        if not sequence.is_floating_point():
            raise ValueError("sequence must have a floating-point dtype")
        dates = self.validate_dates(positions, valid)
        edges = self.edges.to(device=sequence.device, dtype=torch.float64)
        bins = torch.bucketize(dates.contiguous(), edges, right=True) - 1
        index = bins[:, :, None, None, None]
        counts = torch.zeros(b, self.num_bins, 1, h, w, dtype=torch.int64, device=sequence.device)
        counts.scatter_add_(1, index.expand(b, t, 1, h, w), valid.to(torch.int64))

        # where 将无效像素置零，避免 NaN * 0 将 NaN 传播到均值。
        # 低精度特征用 float32 累加，避免均值有限但 half 求和溢出。
        accumulation_dtype = (
            torch.float32 if sequence.dtype in (torch.float16, torch.bfloat16) else sequence.dtype
        )
        values = torch.where(valid, sequence, sequence.new_zeros(())).to(accumulation_dtype)
        sums = torch.zeros(
            b,
            self.num_bins,
            c,
            h,
            w,
            dtype=accumulation_dtype,
            device=sequence.device,
        )
        sums.scatter_add_(1, index.expand(b, t, c, h, w), values)
        pooled = (sums / counts.clamp_min(1).to(accumulation_dtype)).to(sequence.dtype)
        return pooled, counts > 0, counts
