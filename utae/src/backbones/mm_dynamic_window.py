"""窗口均值、按真实日期拟合的一阶趋势及逐像素观测统计。"""

import math

import torch
from torch import nn

from src.backbones.mm_temporal_alignment import normalize_valid_mask


class DynamicWindowRepresentation(nn.Module):
    """日期用 FP64、特征至少用 FP32；无效像素在运算前排除。"""

    def __init__(self, channels=64, hidden=16, min_span=2.0, enabled=True):
        super().__init__()
        if not math.isfinite(min_span) or min_span < 0:
            raise ValueError("min_span must be finite and non-negative")
        self.min_span, self.enabled = float(min_span), bool(enabled)
        self.projection = nn.Sequential(
            nn.Conv2d(channels + 5, hidden, 1), nn.GELU(), nn.Conv2d(hidden, channels, 1)
        )
        self.gamma = nn.Parameter(torch.zeros(()))

    def forward(self, sequence, dates, pad, valid, aligner):
        mean, available, counts = aligner(sequence, dates, pad, valid)
        valid = normalize_valid_mask(sequence, pad, valid)
        dates = aligner.validate_dates(dates.detach(), valid)
        width = (aligner.end_day - aligner.start_day) / aligner.num_bins
        bins = torch.bucketize(dates.contiguous(), aligner.edges, right=True) - 1
        dtype = torch.float64 if sequence.dtype == torch.float64 else torch.float32
        representations, infos, mean_dates, trend_masks = [], [], [], []
        # 逐窗计算，避免展开 [B,T,K,C,H,W]。
        for k in range(aligner.num_bins):
            selected = ((bins == k) & valid.flatten(2).any(-1)).any(0).nonzero(as_tuple=True)[0]
            mask = valid.index_select(1, selected) & (bins.index_select(1, selected) == k)[:, :, None, None, None]
            window_sequence = sequence.index_select(1, selected)
            window_dates = dates.index_select(1, selected)
            n = counts[:, k]
            ok = available[:, k]
            time = window_dates[:, :, None, None, None].expand_as(mask)
            safe_time = torch.where(mask, time, 0)
            tbar = safe_time.sum(1) / n.clamp_min(1)
            lo = time.masked_fill(~mask, float("inf")).amin(1) if window_dates.shape[1] else tbar
            hi = time.masked_fill(~mask, -float("inf")).amax(1) if window_dates.shape[1] else tbar
            span = torch.where(ok, hi - lo, 0)
            delta = torch.where(mask, time - tbar[:, None], 0)
            denominator = delta.square().sum(1)
            trend_ok = ok & (n >= 2) & (span >= self.min_span) & (denominator > 0)
            values = torch.where(mask, window_sequence, 0).to(dtype)
            centered = torch.where(mask, values - mean[:, k, None].to(dtype), 0)
            slope = width * (centered * delta.to(dtype)).sum(1) / denominator.clamp_min(1e-12).to(dtype)
            slope = torch.where(trend_ok, slope, 0)
            info = torch.cat([
                (tbar - aligner.centers[k]) / width,
                n.to(torch.float64).log1p(), span / width,
                ok.to(torch.float64), trend_ok.to(torch.float64),
            ], dim=1)
            info = torch.where(ok, info, 0).to(dtype)
            correction = self.projection(torch.cat([slope, info], 1).to(sequence.dtype))
            result = mean[:, k] + self.gamma * correction if self.enabled else mean[:, k]
            representations.append(torch.where(ok, result, 0))
            infos.append(info)
            mean_dates.append(torch.where(ok, tbar, 0))
            trend_masks.append(trend_ok)
        stats = {
            "info": torch.stack(infos, 1), "mean_dates": torch.stack(mean_dates, 1),
            "trend_valid": torch.stack(trend_masks, 1),
        }
        return torch.stack(representations, 1), available, counts, stats
