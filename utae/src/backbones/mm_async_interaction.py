"""同像素跨模态邻窗交互；真实日期惩罚及有效候选 softmax。"""

import math

import torch
from torch import nn
from torch.nn import functional as F


class LocalAsyncInteraction(nn.Module):
    def __init__(self, modalities=("S2", "S1A", "S1D"), channels=64,
                 dim=16, radius=1, enabled=True):
        super().__init__()
        if isinstance(radius, bool) or not isinstance(radius, int) or radius < 0:
            raise ValueError("radius must be a non-negative integer")
        if isinstance(dim, bool) or not isinstance(dim, int) or dim < 1:
            raise ValueError("interaction dim must be a positive integer")
        self.modalities, self.radius, self.dim = tuple(modalities), radius, dim
        self.enabled = bool(enabled)
        self.query = nn.ModuleDict({m: nn.Conv2d(channels, dim, 1) for m in modalities})
        self.key = nn.ModuleDict({m: nn.Conv2d(channels, dim, 1) for m in modalities})
        self.value = nn.ModuleDict({m: nn.Conv2d(channels, dim, 1) for m in modalities})
        self.output = nn.ModuleDict({m: nn.Conv2d(dim, channels, 1) for m in modalities})
        self.auxiliary = nn.Sequential(nn.Conv2d(11, 8, 1), nn.GELU(), nn.Conv2d(8, 1, 1))
        self.time_penalty = nn.ParameterDict({
            f"{m}_{n}": nn.Parameter(torch.zeros(()))
            for m in modalities for n in modalities if m != n
        })
        self.gamma = nn.ParameterDict({m: nn.Parameter(torch.zeros(())) for m in modalities})

    def forward(self, features, valid, stats, width):
        reference = features[self.modalities[0]]
        b, bins, _, h, w = reference.shape
        zero = reference.new_zeros((), dtype=torch.float32)
        diagnostics = {name: zero.clone() for name in (
            "same_weight", "neighbor_weight", "active_queries", "candidate_count",
            "date_gap_sum", "date_gap_square_sum",
        )}
        diagnostics["date_gap_histogram"] = reference.new_zeros(6, dtype=torch.float32)
        if not self.enabled:
            return features, diagnostics
        mapped = {}
        for m in self.modalities:
            safe = torch.where(valid[m], features[m], 0).reshape(b * bins, -1, h, w)
            mapped[m] = [layer[m](safe).reshape(b, bins, self.dim, h, w)
                         for layer in (self.query, self.key, self.value)]
        results = {}
        for m in self.modalities:
            windows = []
            for k in range(bins):
                candidates, scores, masks = [], [], []
                for n in self.modalities:
                    if n == m:
                        continue
                    for j in range(max(0, k - self.radius), min(bins, k + self.radius + 1)):
                        mask = valid[m][:, k] & valid[n][:, j]
                        gap = ((stats[m]["mean_dates"][:, k] - stats[n]["mean_dates"][:, j])
                               .abs() / width).to(torch.float32)
                        # score/softmax 用 FP32，避免半精度点积溢出。
                        score = (mapped[m][0][:, k].float() * mapped[n][1][:, j].float()).sum(1, keepdim=True)
                        score = score / math.sqrt(self.dim)
                        metadata = torch.cat([stats[m]["info"][:, k], stats[n]["info"][:, j], gap], 1)
                        auxiliary = self.auxiliary(metadata.to(reference.dtype)).float()
                        score = score - F.softplus(self.time_penalty[f"{m}_{n}"]) * gap + auxiliary
                        candidates.append((n, j))
                        scores.append(score)
                        masks.append(mask)
                        with torch.no_grad():
                            observed = gap[mask]
                            diagnostics["candidate_count"] += observed.numel()
                            diagnostics["date_gap_sum"] += observed.sum() * width
                            diagnostics["date_gap_square_sum"] += observed.square().sum() * width**2
                            boundaries = gap.new_tensor([0.25, 0.5, 1., 2., 3.])
                            histogram = torch.bincount(torch.bucketize(observed, boundaries), minlength=6)
                            diagnostics["date_gap_histogram"] += histogram
                mask = torch.stack(masks, 1)
                logits = torch.stack(scores, 1).masked_fill(~mask, torch.finfo(torch.float32).min)
                weights = logits.softmax(1) * mask
                weights = weights / weights.sum(1, keepdim=True).clamp_min(torch.finfo(torch.float32).tiny)
                message = mapped[m][2].new_zeros(b, self.dim, h, w)
                for index, (n, j) in enumerate(candidates):
                    message = message + weights[:, index].to(message.dtype) * mapped[n][2][:, j]
                    key = "same_weight" if j == k else "neighbor_weight"
                    diagnostics[key] += weights[:, index].detach().sum()
                active = mask.any(1)
                diagnostics["active_queries"] += active.sum()
                correction = torch.where(active, self.output[m](message), 0)
                windows.append(torch.where(valid[m][:, k], features[m][:, k] + self.gamma[m] * correction, 0))
            results[m] = torch.stack(windows, 1)
        return results, diagnostics
