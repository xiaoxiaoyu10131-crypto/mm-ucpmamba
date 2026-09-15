"""Quality and late-fusion modules for the independent multimodal UCPMamba."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _resize_quality(quality: torch.Tensor, size):
    if quality.shape[-2:] == tuple(size):
        return quality
    if quality.ndim == 5:
        b, t, c, h, w = quality.shape
        quality = quality.reshape(b * t, c, h, w)
        quality = F.interpolate(quality, size=size, mode="bilinear", align_corners=False)
        return quality.reshape(b, t, c, *size)
    return F.interpolate(quality, size=size, mode="bilinear", align_corners=False)


class FeatureQualityEstimator(nn.Module):
    """Lightweight feature reliability estimator with an optional prior."""

    def __init__(self, in_channels, hidden_channels=None):
        super().__init__()
        hidden_channels = hidden_channels or max(8, in_channels // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=1),
        )

    def forward(self, feature, quality_prior=None):
        sequence = feature.ndim == 5
        if sequence:
            b, t, c, h, w = feature.shape
            flat = feature.reshape(b * t, c, h, w)
        elif feature.ndim == 4:
            b, c, h, w = feature.shape
            t = None
            flat = feature
        else:
            raise ValueError("feature must have shape [B,T,C,H,W] or [B,C,H,W]")

        logits = self.net(flat)
        if quality_prior is not None:
            if sequence and quality_prior.ndim != 5:
                raise ValueError("sequence quality_prior must have shape [B,T,1,H,W]")
            if not sequence and quality_prior.ndim != 4:
                raise ValueError("spatial quality_prior must have shape [B,1,H,W]")
            prior = _resize_quality(quality_prior, (h, w)).to(dtype=logits.dtype)
            if sequence:
                prior = prior.reshape(b * t, 1, h, w)
            prior = prior.clamp(1e-6, 1.0 - 1e-6)
            logits = logits + torch.log(prior) - torch.log1p(-prior)
        quality = torch.sigmoid(logits)
        if sequence:
            return quality.reshape(b, t, 1, h, w)
        return quality


class QualityGuidedPhenologyAttentionV2(nn.Module):
    """PhenologyAttentionV2-compatible skip aggregation with quality bias."""

    def __init__(self, in_channels, stage_level, lambda_q=1.0):
        super().__init__()
        self.stage = stage_level
        self.lambda_q = float(lambda_q)
        out_channels = 64
        if stage_level in [0, 1]:
            self.conv3d = nn.Conv3d(
                in_channels, out_channels, kernel_size=(3, 3, 3), padding=(1, 1, 1)
            )
        else:
            self.conv3d = nn.Conv3d(
                in_channels, out_channels, kernel_size=(3, 1, 1), padding=(1, 0, 0)
            )
        self.attn_net = nn.Sequential(
            nn.ReLU(),
            nn.AdaptiveAvgPool3d((None, 1, 1)),
            nn.Conv3d(out_channels, 1, kernel_size=1),
        )
        self.alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, x, pad_mask, quality=None, return_attention=False):
        b, t, _, _, _ = x.shape
        x_3d = self.conv3d(x.permute(0, 2, 1, 3, 4))
        attn = self.attn_net(x_3d)
        if quality is not None:
            quality = _resize_quality(quality, x.shape[-2:])
            q_t = quality.mean(dim=(-1, -2), keepdim=True).permute(0, 2, 1, 3, 4)
            attn = attn + self.lambda_q * torch.log(q_t.clamp_min(1e-6))
        attn = attn.masked_fill(pad_mask.view(b, 1, t, 1, 1), -1e9)
        attn_weights = F.softmax(attn, dim=2)
        weighted_sum = (x * attn_weights.permute(0, 2, 1, 3, 4)).sum(dim=1)

        mean_mask = (~pad_mask)[:, :, None, None, None].to(dtype=x.dtype)
        mean_sum = (x * mean_mask).sum(dim=1) / mean_mask.sum(dim=1).clamp_min(1.0)
        alpha = torch.sigmoid(self.alpha)
        output = alpha * weighted_sum + (1.0 - alpha) * mean_sum
        if return_attention:
            return output, attn_weights.squeeze(1)
        return output


class LateConcatFusion2D(nn.Module):
    """Per-scale concat + 1x1 projection used by the baseline."""

    def __init__(self, channels, num_modalities=2):
        super().__init__()
        if num_modalities < 2:
            raise ValueError("late fusion requires at least two modalities")
        self.num_modalities = int(num_modalities)
        self.proj = nn.Conv2d(self.num_modalities * channels, channels, kernel_size=1)

    def forward(self, *features, return_weights=False):
        if len(features) != self.num_modalities:
            raise ValueError(
                f"expected {self.num_modalities} modality features, got {len(features)}"
            )
        fused = self.proj(torch.cat(features, dim=1))
        if return_weights:
            return fused, None
        return fused


class QualityAwareFusion2D(nn.Module):
    """Pixel-level quality-aware optical/SAR fusion with an optional residual."""

    def __init__(self, channels, use_complementary_residual=True):
        super().__init__()
        self.use_complementary_residual = bool(use_complementary_residual)
        self.optical_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.sar_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.gate_hidden = nn.Conv2d(4 * channels + 2, channels, kernel_size=3, padding=1)
        self.gate_act = nn.GELU()
        self.gate_logits = nn.Conv2d(channels, 2, kernel_size=1)
        nn.init.zeros_(self.gate_logits.weight)
        nn.init.zeros_(self.gate_logits.bias)
        if self.use_complementary_residual:
            self.complementary = nn.Conv2d(2 * channels, channels, kernel_size=3, padding=1)
            self.gamma = nn.Parameter(torch.zeros(1))
        else:
            self.complementary = None
            self.register_parameter("gamma", None)

    def forward(self, optical, sar, quality_opt=None, quality_sar=None, return_weights=False):
        po = self.optical_proj(optical)
        ps = self.sar_proj(sar)
        qo = optical.new_ones(optical.shape[0], 1, *optical.shape[-2:]) if quality_opt is None else quality_opt
        qs = sar.new_ones(sar.shape[0], 1, *sar.shape[-2:]) if quality_sar is None else quality_sar
        qo = _resize_quality(qo, optical.shape[-2:]).to(dtype=po.dtype).clamp(0.0, 1.0)
        qs = _resize_quality(qs, sar.shape[-2:]).to(dtype=ps.dtype).clamp(0.0, 1.0)
        gate_input = torch.cat([po, ps, (po - ps).abs(), po * ps, qo, qs], dim=1)
        logits = self.gate_logits(self.gate_act(self.gate_hidden(gate_input)))
        weights = torch.softmax(logits, dim=1)
        wo, ws = weights[:, 0:1], weights[:, 1:2]
        adaptive = wo * po + ws * ps
        fused = adaptive
        if self.use_complementary_residual:
            complementary = self.complementary(torch.cat([po, ps], dim=1))
            fused = fused + self.gamma * complementary
        if return_weights:
            return fused, wo, ws
        return fused
